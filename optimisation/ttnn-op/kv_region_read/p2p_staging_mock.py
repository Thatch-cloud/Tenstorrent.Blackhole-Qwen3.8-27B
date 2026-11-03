"""No-device check of the P2P ops (version 4 of the qwen_kv_read extension) on tt-metal's MOCK Blackhole cluster: the real extension, the real ttnn and the real JIT, but no card.

    python3 p2p_staging_mock.py [--cluster-desc <mock cluster descriptor yaml>]

It runs INSIDE a throwaway container of the serving image that has no device mounted (build_kv_read.sh does that) and refuses to run where /dev/tenstorrent exists. The mock
cluster is slow-dispatch and executes nothing, so what it proves is the HOST side of every op against the real tt-metal:

  * the staging allocation and its info (8 banks, 4 devices in the coordinate order of the (4, 1) mesh production opens, aligned 4 KiB fields, one staging per mesh, bad sizes
    refused); the tt-kmd device number of each device is the cluster descriptor's (the mock descriptor permutes them: chip 0 is device 1), not the logical chip id;
  * every kernel is JIT-compiled by tt-metal's own compiler with its own flags the first time it is dispatched (a kernel that does not compile raises QwenP2PEnqueueError,
    which is NOT a refusal: the check breaks each of the three production kernels in turn and sees exactly that), and a missing kernel file is a refusal;
  * the whole refusal matrix (bad bank, offset, stride, block ids, counts, geometry, queue, repeats, tensors that do not match) raises QwenP2PRefused, a ValueError, BEFORE
    anything is enqueued, and a refusal does not disturb the next valid call;
  * the digest handle: ready() / result() shapes, 300 calls with their handles dropped (host buffers outlive the handles), a handle outliving its staging;
  * warm -> freeze: freeze needs warm, a warmed program key keeps working after freeze, a new key is refused; a warm that is refused because of its LAST tensor has built no
    program (every tensor is checked before the first gather); the oracle cannot allocate after warm; a mesh that had a staging frozen takes no new one;
  * the refusal during trace capture (forced with QWEN_P2P_TEST_TRACE_ACTIVE: the mock device cannot capture a trace) for every op.

What it cannot prove: the bytes. The mock device reads back zeros. The kernels' arithmetic is checked on the CPU (test_p2p_staging.py) and on a card (p2p_staging_card.py).

Prints "MOCK-OK <n> checks" and exits 0, or the failed checks and exit 1.
"""

import argparse
import gc
import os
import re
import shutil
import sys
import tempfile
import time

DEFAULT_DESC = 'tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_4xP150.yaml'
BLOCKS = 130                      # blocks of each scratch cache
ROW = 69632                       # one block's row of one tensor on one chip: 64 pages of 1,088 bytes
SHARD = 696320                    # ten tensors


def descriptor_device_numbers(path):
    """`chips_with_mmio` of a cluster descriptor YAML -> {logical chip id: tt-kmd device number}. Read here from the source file, independently of the extension (which reads the
    descriptor tt-metal serializes from the live cluster)."""
    numbers, inside = {}, False
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if not line[0].isspace() and not line.startswith('-'):
                inside = line.split(':')[0].strip() == 'chips_with_mmio'
                continue
            match = re.match(r'\s*-?\s*(\d+)\s*:\s*(\d+)\s*$', line)
            if inside and match:
                numbers[int(match.group(1))] = int(match.group(2))
    return numbers


class QuietFds(object):
    """Sends the C-level stdout and stderr to a file while it is active: tt-metal prints the whole compiler log of a kernel that does not compile, which this check provokes
    on purpose three times and which would bury the build log. The text is kept and shown if a check fails."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.file = tempfile.TemporaryFile(mode='w+b')
        self.saved = (os.dup(1), os.dup(2))
        os.dup2(self.file.fileno(), 1)
        os.dup2(self.file.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        os.close(self.saved[0])
        os.close(self.saved[1])
        self.file.seek(0)
        self.text = self.file.read().decode('utf-8', 'replace')
        self.file.close()
        return False


class Checks(object):
    def __init__(self):
        self.count = 0
        self.failed = []

    def ok(self, condition, what):
        self.count += 1
        if not condition:
            self.failed.append(what)
            print('  FAIL: %s' % what)
        return condition

    def raises(self, exception_type, function, *args, **keywords):
        """-> the exception, after checking that it is exactly of the kind asked for."""
        self.count += 1
        try:
            function(*args, **keywords)
        except exception_type as error:
            return error
        except Exception as error:  # noqa: BLE001
            self.failed.append('%s raised %s: %s' % (getattr(function, '__name__', function), type(error).__name__, str(error)[:160]))
            print('  FAIL: %s' % self.failed[-1])
            return error
        self.failed.append('%s raised nothing (wanted %s)' % (getattr(function, '__name__', function), exception_type.__name__))
        print('  FAIL: %s' % self.failed[-1])
        return None


def run(cluster_desc):
    if os.path.exists('/dev/tenstorrent'):
        print('refusing to run: /dev/tenstorrent exists. This check is for a container with no device mounted.')
        return 2
    os.environ['TT_METAL_MOCK_CLUSTER_DESC_PATH'] = cluster_desc
    os.environ['TT_METAL_SLOW_DISPATCH_MODE'] = '1'
    cache = tempfile.mkdtemp(prefix='p2p_mock_cache_')
    os.environ['TT_METAL_CACHE'] = cache
    os.environ.setdefault('TT_LOGGER_LEVEL', 'ERROR')
    os.environ.pop('QWEN_P2P_KERNEL_DIR', None)
    os.environ.pop('QWEN_P2P_TEST_TRACE_ACTIVE', None)
    import ttnn
    import qwen_kv_read as q

    check = Checks()
    refused, enqueue_error = q.QwenP2PRefused, q.QwenP2PEnqueueError
    started = time.time()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 1))        # as production opens it
    try:
        check.ok(q.QWEN_KV_READ_VERSION == '4', 'the version is 4')
        check.ok(issubclass(refused, ValueError) and not issubclass(refused, RuntimeError), 'QwenP2PRefused is a ValueError (nothing was enqueued)')
        check.ok(issubclass(enqueue_error, RuntimeError) and not issubclass(enqueue_error, ValueError), 'QwenP2PEnqueueError is a RuntimeError (the queue may hold part of the op)')

        def make(shape, count):
            return [ttnn.empty(shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(count)]

        # Ten KV tensors of two shapes with the same block (64 pages of 1,088 bytes): what the Gemma pool has.
        tensors = make([BLOCKS, 1, 128, 512], 5) + make([BLOCKS, 2, 128, 256], 5)
        check.ok(all(int(ttnn.qwen_block_bytes(t)) == ROW for t in tensors), 'every tensor\'s block is 69,632 bytes')

        print('the mesh -> card map, before any staging exists (B1 of the integration review)')
        before = ttnn.qwen_p2p_mesh_devices(mesh)
        check.ok(isinstance(before, list) and len(before) == 4, 'mesh devices: one record per device of the (4, 1) mesh, with no staging allocated')
        check.ok([d['mesh_coord'] for d in before] == [[0, 0], [1, 0], [2, 0], [3, 0]] and [d['mesh_index'] for d in before] == [0, 1, 2, 3], 'mesh devices: in mesh order')
        check.ok(all(d['pcie_device'] is not None and d['pci_bus_id'] is not None and d['pci_bus_id'] == d['pci_bus_id'].lower() for d in before),
                 'mesh devices: every card has its tt-kmd number and a lower case PCI address from the cluster descriptor, without a staging buffer')
        check.ok(len({d['pci_bus_id'] for d in before}) == 4 and len({d['pcie_device'] for d in before}) == 4, 'mesh devices: four distinct cards')
        check.ok(ttnn.qwen_p2p_mesh_devices(mesh) == before, 'mesh devices: a second call answers the same (nothing was built, nothing remembered)')
        check.ok(ttnn.qwen_p2p_mesh_devices.__doc__ and 'NO staging allocation' in ttnn.qwen_p2p_mesh_devices.__doc__, 'mesh devices: the docstring says it needs no staging')
        check.raises(TypeError, ttnn.qwen_p2p_mesh_devices, None)

        print('allocation and info')
        ring = 4 << 20
        staging = ttnn.qwen_p2p_staging_alloc(mesh, ring, False, 4096)
        check.ok(staging.size_bytes == ring and staging.num_banks == 8 and not staging.released, 'alloc: 8 banks of the size asked for')
        info = ttnn.qwen_p2p_staging_info(staging)
        check.ok(len(info['banks']) == 8 and len(info['devices']) == 4, 'info: 8 banks, 4 devices')
        check.ok(all(b['noc_x'] is not None and b['noc_y'] is not None for b in info['banks']), 'info: every bank has NOC coordinates')
        check.ok([d['mesh_coord'] for d in info['devices']] == [[0, 0], [1, 0], [2, 0], [3, 0]], 'info: devices in mesh order, with the coordinates of a (4, 1) mesh')
        check.ok([d['mesh_index'] for d in info['devices']] == [0, 1, 2, 3], 'info: mesh indexes')
        numbers = descriptor_device_numbers(cluster_desc)
        check.ok(len(numbers) == 4, 'the mock cluster descriptor names 4 cards')
        for d in info['devices']:
            check.ok(d['mmio_chip_id'] == d['device_id'], 'info: a P150 is its own MMIO chip (device %s)' % d['device_id'])
            check.ok(d['pcie_device'] == numbers.get(d['mmio_chip_id']), 'info: pcie_device %s is the cluster descriptor\'s tt-kmd device number of chip %s (%s)' % (
                d['pcie_device'], d['mmio_chip_id'], numbers.get(d['mmio_chip_id'])))
            check.ok(d['pci_bus_id'] is None or (isinstance(d['pci_bus_id'], str) and d['pci_bus_id'] == d['pci_bus_id'].lower()), 'info: pci_bus_id is a lower case string or None')
            check.ok(d['pci_bus_id_source'] in ('cluster_descriptor', 'sysfs', None) and (d['pci_bus_id'] is None) == (d['pci_bus_id_source'] is None), 'info: pci_bus_id_source')
            # The mock cluster's addresses are SYNTHETIC (UMD makes them up) and sysfs in the container shows the host's own cards, so the two may well disagree here; on a card
            # they must not (p2p_staging_card.py fails on a disagreement). What is checked is that the fields say what they are: agrees is None when there is nothing to
            # compare, else it is whether the two addresses are equal, and the address reported is the descriptor's whenever the descriptor has one.
            nothing_to_compare = d['pci_bus_id_source'] != 'cluster_descriptor' or d['pci_bus_id_sysfs'] is None
            check.ok((d['pci_bus_id_agrees'] is None) == nothing_to_compare, 'info: pci_bus_id_agrees is None only when there is nothing to compare (device %s)' % d['device_id'])
            check.ok(d['pci_bus_id_agrees'] is None or d['pci_bus_id_agrees'] == (d['pci_bus_id'] == d['pci_bus_id_sysfs']), 'info: pci_bus_id_agrees says whether the addresses are equal')
        check.ok(any(d['pcie_device'] != d['mmio_chip_id'] for d in info['devices']) == (len(set(numbers.items()) - {(c, c) for c in numbers}) > 0),
                 'info: a descriptor that permutes chips and devices gives a pcie_device that is not the chip id (the mock\'s does)')
        check.ok(info['programs_cached'] == 0 and info['pci_map_error'] is None, 'info: nothing built or enqueued yet, and the cluster descriptor was read')
        check.ok(info['devices'] == before, 'info: the devices of the staging info ARE the records the mesh map gave before the allocation (the worker compares the two)')
        print('  devices: %s' % ', '.join('chip %s -> tt-kmd %s (%s, sysfs agrees: %s)' % (d['mmio_chip_id'], d['pcie_device'], d['pci_bus_id_source'], d['pci_bus_id_agrees'])
                                         for d in info['devices']))
        lead = info['aligned_4k_offset']
        check.ok(0 <= lead < 4096 and (info['address'] + lead) % 4096 == 0 and info['aligned_4k_address'] == info['address'] + lead, 'info: the 4 KiB lead')
        check.ok(info['aligned_4k_size'] == (info['size'] - lead) // 4096 * 4096 and info['size'] == ring, 'info: the 4 KiB usable size')
        check.ok(all(b['address'] == info['address'] + (b['bank_offset'] - info['banks'][0]['bank_offset']) for b in info['banks']), 'info: bank addresses')
        check.raises(refused, ttnn.qwen_p2p_staging_alloc, mesh, ring, False, 4096)       # one staging per mesh
        staging.release()
        check.ok(staging.released, 'release')
        check.raises(refused, ttnn.qwen_p2p_staging_alloc, mesh, 1000, False, 4096)       # not a multiple of the page
        check.raises(refused, ttnn.qwen_p2p_staging_alloc, mesh, ring, False, 100)        # a page that is not a multiple of 64
        check.raises(refused, ttnn.qwen_p2p_staging_alloc, mesh, 1 << 40, False, 4096)    # more than a bank can give
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, [1], 64)            # no staging: the digest has nowhere to keep its partials

        print('the first dispatch of each kernel is a real JIT compile')
        staging = ttnn.qwen_p2p_staging_alloc(mesh, ring, False, 4096)
        started_jit = time.time()
        ttnn.qwen_blocks_to_staging(tensors[0], staging, 0, 8192, [3, 1, 2], SHARD, 64)
        ttnn.qwen_write_blocks_from_staging(tensors[0], staging, 0, 8192, [3, 1, 2], SHARD, 64)
        handle = ttnn.qwen_blocks_digest_async(tensors, [3, 1, 2, 7], 64)
        check.ok(handle.blocks == 4 and handle.devices == 4, 'digest handle: 4 blocks on 4 devices')
        check.ok(handle.ready(), 'digest handle: ready (the mock queue completes at once)')
        result = handle.result()
        check.ok(len(result) == 4 and all(len(row) == 4 for row in result) and all(isinstance(v, int) for row in result for v in row), 'digest result: [device][block] ints')
        event = ttnn.qwen_p2p_record_event(mesh)
        check.ok(event.query() is True, 'event: query() is True once the queue is through')
        print('  gather + scatter + digest + event, JIT included: %.1f s' % (time.time() - started_jit))
        sums = ttnn.qwen_staging_checksum(staging, 0, 4096)
        check.ok(isinstance(sums, list) and len(sums) == 4, 'oracle checksum: one value per device')
        ranged = ttnn.qwen_staging_checksum(staging, 0, 4096, 1, 64, 3)
        check.ok(len(ranged) == 4 and all(len(row) == 3 for row in ranged), 'oracle checksum with count: [device][range]')
        check.raises(refused, ttnn.qwen_staging_checksum, staging, 0, 4000)
        check.raises(refused, ttnn.qwen_staging_checksum, staging, 0, 4096, 8)

        print('a refusal never enqueues anything, and the next valid call is unaffected')
        gather = lambda *a, **k: ttnn.qwen_blocks_to_staging(tensors[0], staging, *a, **k)     # noqa: E731
        scatter = lambda *a, **k: ttnn.qwen_write_blocks_from_staging(tensors[0], staging, *a, **k)   # noqa: E731
        cases = [
            ('bank 8', (8, 0, [1], SHARD, 64)),
            ('byte_offset not a multiple of 64', (0, 32, [1], SHARD, 64)),
            ('513 blocks', (0, 0, list(range(513)), SHARD, 64)),
            ('pages_per_block one short', (0, 0, [1], SHARD, 63)),
            ('block id past the pool', (0, 0, [BLOCKS], SHARD, 64)),
            ('no blocks', (0, 0, [], SHARD, 64)),
            ('a stride below the block', (0, 0, [1], ROW - 64, 64)),
            ('a stride that is not a multiple of 64', (0, 0, [1, 2], SHARD + 32, 64)),
            ('a row past the end of the bank range', (0, ring - ROW + 64, [1], SHARD, 64)),
            ('command queue 1 (never CQ1)', (0, 0, [1], SHARD, 64, 1)),
        ]
        for name, args in cases:
            error = check.raises(refused, gather, *args)
            check.ok(error is None or isinstance(error, ValueError), 'gather refusal is a ValueError: ' + name)
            check.raises(refused, scatter, *args)
        check.raises(refused, scatter, 0, 0, [1, 1], SHARD, 64)               # a scatter needs distinct ids ...
        gather(0, 0, [1, 1], SHARD, 64)                                       # ... a gather only reads
        gather(1, 8192, [5, 6], SHARD, 64)
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors + tensors[:7], [1], 64)    # 17 tensors
        check.raises(refused, ttnn.qwen_blocks_digest_async, [], [1], 64)
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, list(range(129)), 64)     # 129 blocks
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, [], 64)
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, [BLOCKS], 64)
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, [1], 63)
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, [1], 64, 1)       # never CQ1
        odd = ttnn.empty([BLOCKS, 1, 64, 512], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)    # 32 pages per block
        check.raises(refused, ttnn.qwen_blocks_digest_async, [tensors[0], odd], [1], 64)       # tensors that do not share a block size
        ttnn.qwen_blocks_digest_async(tensors, [BLOCKS - 1] * 3, 64)                           # (repeats are fine for a read)
        check.raises(refused, ttnn.qwen_p2p_freeze, staging)                                   # nothing warmed yet
        check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors, BLOCKS)                    # the null block is outside
        check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors, 0, 9)                      # a bank that does not exist
        check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors, 0, 0, ring - 1000)         # a scratch that does not fit the bank's range

        print('the refusal during trace capture (forced: the mock device cannot capture a trace)')
        os.environ['QWEN_P2P_TEST_TRACE_ACTIVE'] = '1'
        try:
            for name, function in (('gather', lambda: gather(0, 0, [1], SHARD, 64)), ('scatter', lambda: scatter(0, 0, [1], SHARD, 64)),
                                   ('digest', lambda: ttnn.qwen_blocks_digest_async(tensors, [1], 64)), ('event', lambda: ttnn.qwen_p2p_record_event(mesh)),
                                   ('warm', lambda: ttnn.qwen_p2p_warm(staging, tensors)), ('oracle', lambda: ttnn.qwen_staging_checksum(staging, 0, 4096)),
                                   ('alloc', lambda: ttnn.qwen_p2p_staging_alloc(mesh, ring, False, 4096))):
                error = check.raises(refused, function)
                check.ok(error is not None and 'trace' in str(error), 'refusal during capture says why: ' + name)
        finally:
            del os.environ['QWEN_P2P_TEST_TRACE_ACTIVE']
        gather(0, 0, [1], SHARD, 64)

        print('kernels that cannot be found or do not compile')
        good_dir = os.path.dirname(os.path.abspath(q.__file__))
        empty, broken = tempfile.mkdtemp(prefix='p2p_mock_empty_'), tempfile.mkdtemp(prefix='p2p_mock_broken_')
        try:
            for name in ('qwen_p2p_gather.cpp', 'qwen_p2p_scatter.cpp', 'qwen_p2p_digest_rows.cpp'):
                with open(os.path.join(broken, name), 'w') as handle_file:
                    handle_file.write('#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() { this_is_not_a_function(); }\n')
            for label, directory, kind in (('missing', empty, refused), ('broken', broken, enqueue_error)):
                staging.release()
                staging = ttnn.qwen_p2p_staging_alloc(mesh, ring, False, 4096)
                os.environ['QWEN_P2P_KERNEL_DIR'] = directory
                failed_before = len(check.failed)
                try:
                    with QuietFds() as quiet:
                        errors = [check.raises(kind, lambda: ttnn.qwen_blocks_to_staging(tensors[0], staging, 0, 8192, [1], SHARD, 64)),
                                  check.raises(kind, lambda: ttnn.qwen_write_blocks_from_staging(tensors[0], staging, 0, 8192, [1], SHARD, 64)),
                                  check.raises(kind, lambda: ttnn.qwen_blocks_digest_async(tensors, [1], 64))]
                finally:
                    del os.environ['QWEN_P2P_KERNEL_DIR']
                if label == 'broken':
                    for error in errors:
                        check.ok('Failed to generate binar' in str(error) or 'this_is_not_a_function' in quiet.text + str(error),
                                 'tt-metal\'s own JIT compiler reported the broken kernel (it really compiled it): ' + str(error)[:200])
                if len(check.failed) > failed_before:
                    print(quiet.text[-3000:])
                print('  %s kernels: %s' % (label, kind.__name__))
            check.ok(os.path.exists(os.path.join(good_dir, 'qwen_p2p_gather.cpp')), 'the real kernels ship beside the module')
        finally:
            shutil.rmtree(empty, True)
            shutil.rmtree(broken, True)

        print('warm, freeze, and a program key that was not warmed')
        staging.release()
        staging = ttnn.qwen_p2p_staging_alloc(mesh, ring, False, 4096)
        # A warm that is refused enqueued nothing, whichever tensor is the bad one: all are checked before the first gather (a bad LAST tensor is the case that used to run nine
        # gathers first). No program was built either: programs_cached counts the builds, and the first gather builds one.
        short = ttnn.empty([4, 1, 128, 512], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)   # 4 blocks: no block 7
        error = check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors[:9] + [short], 7)
        check.ok(error is not None and 'tensor 9' in str(error), 'a warm refused for its last tensor says which tensor: ' + str(error)[:160])
        error = check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors[:9] + [odd])                    # a block of 32 pages in the last place
        check.ok(error is not None and 'tensor 9' in str(error), 'a warm refused for the block size of its last tensor says which tensor: ' + str(error)[:160])
        check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors, 0, 0, ring - 1000)
        check.ok(ttnn.qwen_p2p_staging_info(staging)['programs_cached'] == 0 and not staging.warmed, 'a refused warm built no program and warmed nothing')
        ttnn.deallocate(short)
        started_warm = time.time()
        ttnn.qwen_p2p_warm(staging, tensors)
        check.ok(staging.warmed and not staging.frozen, 'warm')
        error = check.raises(refused, ttnn.qwen_staging_checksum, staging, 0, 4096)         # warmed but not yet frozen: the oracle still allocates nothing
        check.ok(error is not None and 'before warm' in str(error), 'the oracle allocates nothing after warm, even before freeze: ' + str(error)[:160])
        ttnn.qwen_p2p_freeze(staging)
        check.ok(staging.frozen, 'freeze')
        ttnn.qwen_p2p_freeze(staging)                                                       # idempotent
        print('  warm + freeze, JIT included: %.1f s' % (time.time() - started_warm))
        gather(2, 8192, [5, 6], SHARD, 64)                                                  # a warmed key keeps working
        scatter(2, 8192, [5, 6], SHARD, 64)
        ttnn.qwen_blocks_digest_async(tensors, [5, 6], 64)
        ttnn.qwen_blocks_digest_async(tensors[:3], [5, 6], 64)                              # (the tensor count is a runtime argument, not a new key)
        error = check.raises(refused, ttnn.qwen_blocks_to_staging, odd, staging, 0, 8192, [1], 34816, 32)    # a block of 32 pages: a new key
        check.ok(error is not None and 'not warmed' in str(error), 'a new program key after freeze is refused, and says why')
        check.raises(refused, ttnn.qwen_p2p_warm, staging, tensors)                         # warm after freeze
        error = check.raises(refused, ttnn.qwen_staging_checksum, staging, 0, 4096)         # the oracle's partials were not allocated before warm: no allocation now
        check.ok(error is not None and 'before warm' in str(error), 'the oracle allocates nothing after warm, and says why: ' + str(error)[:160])
        check.ok(ttnn.qwen_p2p_staging_info(staging)['programs_cached'] >= 3 and staging.frozen, 'the warmed programs are cached')

        print('digest handles and host buffers')
        started_calls = time.time()
        kept = []
        for i in range(300):
            kept.append(ttnn.qwen_blocks_digest_async(tensors, [1 + (i % 100), 2, 3], 64))
            if len(kept) > 5:
                kept.pop(0)
        check.ok(kept[-1].ready(), '300 digests in a row')
        print('  300 digest calls: %.2f s' % (time.time() - started_calls))
        del kept
        gc.collect()
        straggler = ttnn.qwen_blocks_digest_async(tensors, [1, 2], 64)
        staging.release()
        check.ok(straggler.ready() and len(straggler.result()) == 4, 'a handle outlives its staging and stays readable')
        check.raises(refused, ttnn.qwen_blocks_digest_async, tensors, [1], 64)
        check.raises(refused, ttnn.qwen_blocks_to_staging, tensors[0], staging, 0, 0, [1], SHARD, 64)
        error = check.raises(refused, ttnn.qwen_p2p_staging_alloc, mesh, ring, False, 4096)    # a mesh that had a staging frozen takes no new one, though that one is released
        check.ok(error is not None and 'frozen' in str(error), 'a mesh that had a staging frozen takes no new staging (traces may exist), and the refusal says why: ' + str(error)[:160])
        check.ok(ttnn.qwen_p2p_mesh_devices(mesh) == before, 'mesh devices: after the staging was frozen and released the map is the same, and the call still works')
    finally:
        ttnn.close_mesh_device(mesh)
        shutil.rmtree(cache, True)
    error = check.raises(refused, ttnn.qwen_p2p_mesh_devices, mesh)
    check.ok(error is not None and 'closed' in str(error), 'mesh devices: a closed mesh is a refusal that says so: ' + str(error)[:120])
    print('%d checks in %.1f s' % (check.count, time.time() - started))
    if check.failed:
        print('MOCK-FAILED %d of %d checks:' % (len(check.failed), check.count))
        for what in check.failed:
            print('  - %s' % what)
        return 1
    print('MOCK-OK %d checks' % check.count)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    home = os.environ.get('TT_METAL_HOME', '/opt/tt-metal')
    parser.add_argument('--cluster-desc', default=os.environ.get('QWEN_MOCK_CLUSTER_DESC') or os.path.join(home, DEFAULT_DESC), help='a 4-chip Blackhole mock cluster descriptor')
    options = parser.parse_args(argv)
    return run(options.cluster_desc)


if __name__ == '__main__':
    sys.exit(main())
