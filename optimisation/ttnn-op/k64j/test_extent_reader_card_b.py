"""CPU checks for W10b's CB2b harness (extent_reader_card_b.py; run_card_b.sh K64J_HARNESS=extent_reader); no device.

  - the helpers: the live starts (the floor, the table's end), the family plan (the named first, more than 50 by
    default), the replay plan (every family, the restages), the idle patterns, the decision, the scope and the
    verdict line, the PINDIAG check, the arguments;
  - the contract: the pinned shas are the served ones (test_extent_attention_replay's), and the runner neither ships
    nor checks a checkout copy of them; the lent storage is what serving_buffer_pool's extent storage is, tensor for
    tensor; the two host mirrors of the mask kernel agree; the two-chip view (one shard twice, chip 0's program
    launched, chip 1's counted, sys.modules restored); without the view the real reader refuses a one-chip device;
  - the served loader, in a fresh interpreter as the container runs it: the pinned modules from a served tree rebuilt
    from git (the frozen recipe's stage of the mask module), the code under test bound to them, and a served tree
    with other bytes, a missing module or one imported first refused;
  - the runner (needs bash): K64J_HARNESS=extent_reader's dry run (the harness and this checkout's scripts/ci mounted
    read-only, the image's pinned sources logged, QWEN_FAST_SDPA_MODES, the report and container names), its watcher
    pass, an unknown harness and a missing source refused before anything is launched, card M through the override
    (the dry run, and the real launch path on test_qual_card's fake rig), and the cardm job file's check of the
    documented values;
  - the whole flow on a fake ONE-chip ttnn (test_k64j_card_b.FakeExtentTtnn's K64j SDPA, read at replay time, plus
    the pinned mask and fold kernels emulated from their .cpp, slices, concats and traces): the REAL reader classes
    through the two-chip view, PASS end to end, and each broken variant on the section that must catch it - a stale
    cur_pos, an absolute word, a wide mask read, a missing slot copy (alone, and with a device that lacks R10's slot
    copy), an unstaged construction, a mask kernel that skips a tile, a trace that keeps its captured cur_pos, a
    non-finite idle row, a missing PINDIAG or F22 line, a module from elsewhere, the wrong QWEN_FAST_SDPA_MODES and
    the deadline; and the ones that only one part of the harness can see - a replayed mask launch on its captured word
    (only a real replay in R1), one that skips a tile whose bits the eager run left right (only the NaN poison), an
    idle segment that moves its live neighbour (only r4_live_unchanged), or a live one at C (only R4 at the
    assignment holding C), a device that finds E without reading cur_pos (only cur_pos_live: NO-DECISION, never
    PASS) or that ignores every mask (only mask_live).

    py -3.11 -B -m unittest test_extent_reader_card_b      (from this directory; scripts/ci on the path)
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
PROBE_DIR = OPS / 'k64j_probe'
for _path in (str(HERE), str(PROBE_DIR), str(OPS / 'sdpa_decode_qwen'), str(CI)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import attention_fold_dma_tp  # noqa: E402
import chip_view  # noqa: E402
import extent_attention_replay as extent_module  # noqa: E402
import extent_attention_replay_tp as extent_tp  # noqa: E402
import packed_any_admission as admission  # noqa: E402
import record_packed_any_evidence_tp4 as recorder  # noqa: E402
import pooled_attention_replay  # noqa: E402
import extent_reader_card_b as reader_b  # noqa: E402
import k64j_card_b as card_b  # noqa: E402
import split_model as model  # noqa: E402
import test_k64j_card_b as card_tests  # noqa: E402 - FakeExtentTtnn, make_graft, the runner's fake rig
import test_k64j_probe as probe_tests  # noqa: E402 - FakeTensor, HostTensor, ELEMENT, bash

probe = card_b.probe
card = probe.card
RUNNER = HERE / 'run_card_b.sh'
CARD_B, CARD_M, CARD_A = probe_tests.CARD_B, probe_tests.CARD_M, card_tests.qual_tests.CARD_A
CARD_X = probe_tests.CARD_X
BASH = probe_tests.BASH
SCRUB = probe_tests.SCRUB + ('K64J_CARD_DRY_RUN', 'K64J_HARNESS', 'QWEN_FAST_SDPA_MODES', 'TP4_WIDTH', 'QWEN_FAST_TP')
NL = chr(10)
ONE = ('range', ('coord', 0, 0), ('coord', 0, 0))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


def checkout_pinned():
    """reader_b.PINNED as this checkout holds those files. The in-process flow takes the pinned modules from sys.path
    (--served-root ''), and those are the checkout's copies, whose mask module is not the served one.
    ServedLoaderTests loads a served tree in a fresh interpreter."""
    import test_extent_attention_replay as extent_tests
    return {name: extent_tests.StructureTests.CHECKOUT[name] for name in reader_b.PINNED}


# ---------------------------------------------------------------------------------------------
# A fake ONE-chip ttnn: the K64j SDPA of test_k64j_card_b.FakeExtentTtnn, read at replay time, and the pinned mask and
# fold kernels emulated from their .cpp.
# ---------------------------------------------------------------------------------------------

class Shard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class ReaderTensor(probe_tests.FakeTensor):
    def __init__(self, address, shape, dtype, nbytes, layout, memory):
        super().__init__(address, shape, dtype, nbytes)
        self.layout, self.memory = layout, memory

    @property
    def padded_shape(self):
        return self.shape

    def memory_config(self):
        return self.memory


class RuntimeArgs(dict):
    def __missing__(self, key):
        value = self[key] = {}
        return value


class MeshProgram(dict):
    pass


class FakeReaderTtnn(card_tests.FakeExtentTtnn):
    """One chip: get_device_tensors gives one shard and generic_op takes a program for mesh coordinate (0, 0) only.
    generic_op runs the pinned kernels from their runtime arguments (addresses included): attention_mask_replay.cpp
    (the task's 32 x 32 tile at its page, -inf where the cache position is past the row's) and attention_fold_dma.cpp
    (the tile permutation, forward and inverse). Slices, concats, the kernels and the K64j SDPA read device memory by
    address when they run, so a trace replays them on what is staged at replay time. The K64j SDPA and the 0x7
    reference are test_k64j_card_b.FakeExtentTtnn's (the split, the tree, bf16 rounding, the tail mask, share on
    slot 0).

    broken (besides FakeExtentTtnn's 'own_slot', 'stale_trace', 'tail_at_capacity', 'no_f22'): 'mask_stale_tile' (the
    mask kernel never writes column tile 7), 'idle_nonfinite' (an entry at word 255 on an all-zero table comes back
    NaN), 'mask_trace_word' (a mask launch replayed from a trace reads the word its positions tensor held at capture,
    not at replay), 'mask_trace_skips_tile' (a replayed mask launch never writes column tile 0; an eager one does),
    'idle_bleeds' (an idle entry - word 255 on an all-zero table - moves the output of the live segments next to its
    own, in the same forward), 'idle_bleeds_at_capacity' (it moves every live segment at E = C in the same forward),
    'extent_from_table' (a 0x20 program ignores its cur_pos_tensor and takes E from the table instead: the first
    poison page, else C - right on every poisoned table, so only the cur_pos liveness control can tell), 'ignore_mask'
    (every call, 0x20 or not, ignores its attention mask)."""

    class DataMovementProcessor:
        RISCV_0 = 'riscv0'

    class NOC:
        RISCV_0_default = 'noc0'

    head_rows = 12                          # query-head rows per token the kernels are built for (6 at width 4: the define)

    def __init__(self, torch, *args, **kwargs):
        super().__init__(torch, *args, **kwargs)
        self.launched = []
        self.forward = []                   # (idle, live at C) per 0x20 call run since the block's last concat

    # tensors --------------------------------------------------------------------------------
    def allocate(self, shape, dtype, layout=None, memory=None):
        nbytes = int(self.torch.Size(tuple(shape)).numel()) * probe_tests.ELEMENT[dtype]
        pool = self.free.get(nbytes) if self.reuse else None
        if pool:
            address = pool.pop()
        else:
            address, self.next = self.next, self.next + nbytes
        return ReaderTensor(address, tuple(shape), dtype, nbytes, self.TILE_LAYOUT if layout is None else layout,
                            self.DRAM_MEMORY_CONFIG if memory is None else memory)

    def from_torch(self, host, *, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        if device is None:
            return probe_tests.HostTensor(host.clone())
        tensor = self.allocate(host.shape, dtype, layout, memory_config)
        self.memory[tensor.address] = host.clone()
        return tensor

    @staticmethod
    def ReplicateTensorToMesh(mesh):  # noqa: N802
        return ('replicate', id(mesh))

    def get_device_tensors(self, tensor):
        return [Shard(tensor.address)]

    def empty(self, shape, dtype, layout, device, memory_config):
        tensor = self.allocate(tuple(shape), dtype, layout, memory_config)
        self.memory[tensor.address] = self.torch.zeros(tuple(shape), dtype=self.torch.bfloat16)
        return tensor

    def run_or_record(self, write):
        if self.capturing is not None:
            self.traces[self.capturing].append(write)
        else:
            write()

    def slice(self, tensor, start, stop, memory_config=None):
        index = tuple(slice(a, b) for a, b in zip(start, stop))
        output = self.allocate(tuple(b - a for a, b in zip(start, stop)), tensor.dtype, self.TILE_LAYOUT, memory_config)
        source = tensor.address

        def write(replay=False):
            self.memory[output.address] = self.memory[source][index].clone()

        self.run_or_record(write)
        return output

    def concat(self, parts, dim, memory_config=None):
        shape = list(parts[0].shape)
        shape[dim] = sum(part.shape[dim] for part in parts)
        output = self.allocate(tuple(shape), parts[0].dtype, self.TILE_LAYOUT, memory_config)
        sources = [part.address for part in parts]
        block = dim == 1 and len(parts) == len(reader_b.SEGMENTS) and shape[1] == reader_b.SEGMENTS[-1][1]

        def write(replay=False):
            data = self.torch.cat([self.memory[address] for address in sources], dim=dim)
            self.memory[output.address] = self.bleed(data) if block else data

        self.run_or_record(write)
        return output

    def bleed(self, data):
        """The block's own concat ends a forward: its four 0x20 calls ran since the last one, in segment order.
        'idle_bleeds' moves (+1) the live segments next to an idle one; 'idle_bleeds_at_capacity' every live segment
        at E = C while any segment is idle."""
        calls, self.forward = self.forward, []
        if len(calls) != len(reader_b.SEGMENTS) or not any(idle for idle, _ in calls):
            return data
        moved = set()
        for index, (idle, at_capacity) in enumerate(calls):
            if idle:
                continue
            if 'idle_bleeds' in self.broken and any(calls[other][0] for other in (index - 1, index + 1)
                                                    if 0 <= other < len(calls)):
                moved.add(index)
            if 'idle_bleeds_at_capacity' in self.broken and at_capacity:
                moved.add(index)
        for index in sorted(moved):
            first, last = reader_b.SEGMENTS[index]
            data[:, first:last] = (data[:, first:last].float() + 1).to(data.dtype)
        return data

    # program descriptors ----------------------------------------------------------------------
    def CoreCoord(self, x, y):  # noqa: N802
        return (x, y)

    def CoreRange(self, first, last):  # noqa: N802
        return (first, last)

    def CoreRangeSet(self, ranges):  # noqa: N802
        return list(ranges)

    def CBDescriptor(self, **options):  # noqa: N802
        return SimpleNamespace(**options)

    def CBFormatDescriptor(self, **options):  # noqa: N802
        return SimpleNamespace(**options)

    def TileDescriptor(self, tile):  # noqa: N802
        return tile

    def Tile(self, shape):  # noqa: N802
        return tuple(shape)

    def TensorAccessorArgs(self, shard):  # noqa: N802
        return SimpleNamespace(get_compile_time_args=lambda: [shard.buffer_address() & 0xff])

    def DataMovementConfigDescriptor(self, **options):  # noqa: N802
        return SimpleNamespace(**options)

    def KernelDescriptor(self, **options):  # noqa: N802
        return SimpleNamespace(runtime_args=None, **options)

    def RuntimeArgs(self):  # noqa: N802
        return RuntimeArgs()

    def ProgramDescriptor(self, **options):  # noqa: N802
        return SimpleNamespace(**options)

    def MeshCoordinate(self, row, column):  # noqa: N802
        return ('coord', row, column)

    def MeshCoordinateRange(self, first, last):  # noqa: N802
        return ('range', first, last)

    def MeshProgramDescriptor(self):  # noqa: N802
        return MeshProgram()

    def generic_op(self, tensors, program):
        if not isinstance(program, MeshProgram) or set(program) != {ONE}:
            raise RuntimeError('TT_FATAL: a mesh program for devices outside the 1x1 mesh: %r'
                               % (sorted(program) if isinstance(program, dict) else program,))
        launches = []
        shapes = {tensor.address: tuple(tensor.shape) for tensor in tensors}
        for kernel in program[ONE].kernels:
            path = Path(kernel.kernel_source)
            if not path.is_file():
                raise RuntimeError('TT_FATAL: kernel source %s not found' % path)
            tasks = [list(values) for column in kernel.runtime_args.values() for values in column.values()]
            run = {'attention_mask_replay.cpp': self.mask_launch, 'attention_fold_dma.cpp': self.fold_launch,
                   'attention_mask_replay_tp.cpp': self.mask_launch,
                   'attention_fold_dma_tp.cpp': self.fold_launch}.get(path.name)
            if run is None:
                raise RuntimeError('TT_FATAL: no emulation of %s' % path.name)
            # The sibling kernels take the head-row count as a define, the pinned ones as a literal and no defines.
            defines = dict(getattr(kernel, 'defines', None) or ())
            want = {'QWEN_FOLD_HEAD_ROWS': str(self.head_rows)} if path.name.endswith('_tp.cpp') else {}
            if defines != want:
                raise RuntimeError('TT_FATAL: %s built with defines %r, not %r' % (path.name, defines, want))
            # What each positions word holds as the launch is issued (at capture, for a launch recorded in a trace).
            seen = ({task[0]: self.memory[task[0]].clone() for task in tasks}
                    if path.name == 'attention_mask_replay.cpp' else {})
            launches.append((run, tasks, seen))
            self.launched.append((path.name, len(tasks)))

        def write(replay=False):
            for run, tasks, seen in launches:
                run(tasks, shapes, replay, seen)

        self.run_or_record(write)

    def mask_launch(self, tasks, shapes, replay=False, seen=None):
        """attention_mask_replay.cpp, per task: its (batch, head tile, column tile), the word read from the positions
        tensor, the tile of 0 / -inf, written at page (b * HT + ht) * (capacity / 32) + capacity / 32 - 8 + ct."""
        torch = self.torch
        target = tasks[0][1]
        mask = self.memory[target].clone()
        if tuple(mask.shape) != shapes[target]:
            raise RuntimeError('the mask changed shape: a reader-owned mask is never freed')
        batches, _, heads, width = mask.shape
        mask_row_tiles = (heads + 31) // 32
        for positions, destination, rows, capacity, offset, task in tasks:
            if destination != target:
                raise RuntimeError('one mask per launch')
            source = seen if (replay and 'mask_trace_word' in self.broken) else self.memory
            word = int(source[positions].reshape(-1)[0]) & 0xffffffff
            head_tiles = (rows * self.head_rows + 31) // 32
            batch, head_tile, column_tile = task // (head_tiles * 8), (task // 8) % head_tiles, task % 8
            if 'mask_stale_tile' in self.broken and column_tile == 7:
                continue
            if replay and 'mask_trace_skips_tile' in self.broken and column_tile == 0:
                continue
            head = head_tile * 32 + torch.arange(32)
            position = word + offset + batch * rows + (head % (rows * 6)) // 6
            cache = capacity - 256 + column_tile * 32 + torch.arange(32)
            masked = (head[:, None] >= rows * self.head_rows) | (cache[None, :] > position[:, None])
            tile = torch.where(masked, float('-inf'), 0.0).to(torch.bfloat16)
            page = (batch * head_tiles + head_tile) * (capacity // 32) + capacity // 32 - 8 + column_tile
            block, column = divmod(page, width // 32)
            entry, row_tile = divmod(block, mask_row_tiles)
            if entry >= batches:
                raise RuntimeError('TT_FATAL: mask kernel page %d is past the %r mask' % (page, tuple(mask.shape)))
            first, last = row_tile * 32, min(row_tile * 32 + 32, heads)
            mask[entry, 0, first:last, column * 32:(column + 1) * 32] = tile[:last - first]
        self.memory[target] = mask

    def fold_launch(self, tasks, shapes, replay=False, seen=None):
        """attention_fold_dma.cpp: task t writes output tile t (its row tile t / 8, column tile t % 8). Forward,
        folded row h of the output is token offset + (h % (rows * 6)) / 6, head (h / (rows * 6)) * 6 + h % 6 of the
        source; inverse, token t's head r is folded row (r / 6) * rows * 6 + t * 6 + r % 6. Every task writes its
        whole tile, so the output starts from nothing: its address may have held another tensor of the same size (a
        freed intermediate reused inside the trace). Only the tiles of the launched tasks are written."""
        torch = self.torch
        source, target, rows, inverse, offset = tasks[0][:5]
        if any(tuple(task[:5]) != (source, target, rows, inverse, offset) for task in tasks):
            raise RuntimeError('one source, output and geometry per launch')
        data = self.memory[source]
        if tuple(data.shape) != shapes[source]:
            raise RuntimeError('fold source %r holds %r at run time' % (shapes[source], tuple(data.shape)))
        output = torch.zeros(shapes[target], dtype=torch.bfloat16)
        if inverse:
            covered = torch.zeros(rows, 8, dtype=torch.bool)
            for task in tasks:
                covered[task[5] // 8, task[5] % 8] = True
            heads = torch.arange(self.head_rows)
            folded = (heads // 6)[None, :] * rows * 6 + torch.arange(rows)[:, None] * 6 + (heads % 6)[None, :]
            full = data[0, 0][folded]                                            # (rows, head_rows, 256)
            cover = covered.repeat_interleave(32, dim=1)[:, None, :].expand(rows, self.head_rows, 256)
            output[0] = torch.where(cover, full, output[0])
        else:
            count = rows * self.head_rows
            covered = torch.zeros((count + 31) // 32, 8, dtype=torch.bool)
            for task in tasks:
                covered[task[5] // 8, task[5] % 8] = True
            heads = torch.arange(count)
            remainder = heads % (rows * 6)
            full = data[0, remainder // 6 + offset, (heads // (rows * 6)) * 6 + remainder % 6]     # (count, 256)
            cover = covered.repeat_interleave(32, dim=0)[:count].repeat_interleave(32, dim=1)
            output[0, 0] = torch.where(cover, full, output[0, 0])
        self.memory[target] = output

    # the K64j SDPA, read when it runs -----------------------------------------------------------
    def extent_call(self, query, k, v, pages, scale, k_chunk, flags, attn_mask, cur_pos_tensor):
        torch = self.torch
        batches, rows = query.shape[1], query.shape[2]
        capacity = pages.shape[1] * card.PAGE
        share = bool(flags & card_b.SHARE) and batches > 1
        key = (flags, batches, capacity // 32, 8)
        if not self.silent and key not in self.programs:
            self.programs.add(key)
            os.write(1, ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=8 kv_share=%s '
                         'scratch_slots=4 cb_bytes=7\n' % (flags, batches, -(-rows // 32), capacity // 32,
                                                           'true' if share else 'false')).encode())
            if 'no_f22' not in self.broken:
                os.write(1, ('Op | INFO | [QWEN-SDPA] runtime-extent entries=%d kv_share=%s q_slice=%s '
                             'writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes=64\n'
                             % (batches, 'true' if share else 'false',
                                'true' if flags & card_b.SLICE else 'false')).encode())
        self.calls += 1
        cores = model.cores_per_head(batches)
        captured = self.memory[cur_pos_tensor.address].clone()
        output = self.allocate((1, batches, rows, card.HEAD_DIM), 'bf16')
        self.memory[output.address] = torch.zeros(output.shape, dtype=torch.bfloat16)

        def write(replay=False):
            q = self.memory[query.address].float()
            table = self.memory[pages.address]
            keys, values = self.read_float(k), self.read_float(v)
            current = captured if (replay and 'stale_trace' in self.broken) else self.memory[cur_pos_tensor.address]
            words = [int(value) for value in current.reshape(-1).tolist()]
            mask = self.mask_seen(self.memory[attn_mask.address], flags, batches, capacity)
            if 'ignore_mask' in self.broken:
                mask = torch.zeros_like(mask)
            data = torch.zeros(output.shape, dtype=torch.bfloat16)
            idle, live_at_capacity = [], False
            for entry in range(batches):
                word = words[entry if (not share or 'own_slot' in self.broken) else 0]
                row = table[0 if share else entry]
                if 'extent_from_table' in self.broken:
                    poisoned = (row >= keys.shape[0] - probe.POISON_BLOCKS).nonzero()
                    word = (int(poisoned[0]) * card.PAGE if len(poisoned) else capacity) - 1
                idle.append(word == 255 and not bool(row.any()))
                live_at_capacity = live_at_capacity or (not idle[-1] and word == capacity - 1)
                if 'idle_nonfinite' in self.broken and idle[-1]:
                    data[0, entry] = float('nan')
                    continue
                result = self.attend(q[0, entry], keys, values, row, word, k_chunk // 32, scale, cores, False,
                                     mask[entry, 0], True, capacity)
                data[0, entry] = result.to(torch.bfloat16)
            self.memory[output.address] = data
            self.forward.append((all(idle), live_at_capacity))

        self.run_or_record(write)
        self.now += 20e-6 + 1e-9 * capacity if self.seconds_per_call is None else self.seconds_per_call
        return output

    def paged_scaled_dot_product_attention_decode(self, query, k, v, *positional, **options):
        """'ignore_mask' on the non-extent calls (the 0x7 references): their mask read as zeros."""
        mask = options.get('attn_mask')
        if 'ignore_mask' in self.broken and mask is not None and options.get('cur_pos_tensor') is None:
            options['attn_mask'] = self.from_torch(self.torch.zeros_like(self.read(mask)), dtype=self.bfloat16,
                                                   layout=self.TILE_LAYOUT, device=self)
        return super().paged_scaled_dot_product_attention_decode(query, k, v, *positional, **options)


# ---------------------------------------------------------------------------------------------
# The helpers.
# ---------------------------------------------------------------------------------------------

def parse(argv=()):
    return reader_b.parse_args(['--out', 'x.json'] + list(argv))


class HelperTests(unittest.TestCase):
    def test_live_starts_stay_in_their_family_above_the_floor_and_inside_the_table(self):
        capacity = reader_b.CAPACITY
        for family in (256, 512, 2304, 131072, capacity):
            for residue in range(256):
                start = reader_b.live_start(family, residue, capacity)
                self.assertEqual(model.extent(start), family)
                self.assertTrue(start >= 128 and start + 16 <= capacity, (family, residue, start))
        self.assertEqual([reader_b.live_start(256, residue, capacity) for residue in (0, 7, 127, 240, 255)],
                         [128, 135, 255, 240, 255])
        self.assertEqual(reader_b.live_start(capacity, 255, capacity), capacity - 16)
        self.assertEqual(reader_b.live_start(2304, 7, capacity), 2055)

    def test_the_default_family_plan_is_more_than_50_with_the_named_first(self):
        families = reader_b.family_plan(reader_b.CAPACITY, reader_b.R2_NAMED, reader_b.R2_FAMILIES)
        self.assertEqual(families[:5], list(reader_b.R2_NAMED))
        self.assertEqual(len(set(families)), 56)
        self.assertGreater(len(families), 50)
        self.assertTrue(all(family % 256 == 0 and 256 <= family <= reader_b.CAPACITY for family in families))
        self.assertIn(512, families)
        self.assertLess(min(families[5:]), 4096)                     # the stale-writer zone is spread into too
        self.assertEqual(reader_b.family_plan(4352, reader_b.R2_NAMED, 2), [256, 2304])
        self.assertEqual(reader_b.family_plan(4352, reader_b.R2_NAMED, 3), [256, 2304, 2560])
        self.assertEqual(reader_b.family_plan(4352, (256, 512, 2304, 4352), 6), [256, 512, 2304, 4352, 768, 4096])
        with self.assertRaises(ValueError):
            reader_b.family_plan(1024, (256,), 10)

    def test_the_replay_plan_visits_every_family_and_restages_twice(self):
        families = reader_b.family_plan(reader_b.CAPACITY, reader_b.R2_NAMED, reader_b.R2_FAMILIES)
        plan = reader_b.replay_plan(families, reader_b.DESIGN_RESIDUES, 2, reader_b.CAPACITY)
        self.assertEqual(len(plan), 28)
        self.assertEqual(plan[0]['families'], [256, 2304, 16640, 65792])
        self.assertEqual({family for entry in plan for family in entry['families']}, set(families))
        self.assertEqual([entry['tables'] for entry in plan[:4]], [True, False, True, False])
        for entry in plan:
            for family, start in zip(entry['families'], entry['starts']):
                self.assertEqual(model.extent(start), family)
                self.assertTrue(extent_module.admits(start, 16, reader_b.CAPACITY), start)
        for first, second in zip(plan[::2], plan[1::2]):
            self.assertEqual(first['families'], second['families'])
            self.assertNotEqual(first['starts'], second['starts'])
        residues = {start % 256 for entry in plan for start, family in zip(entry['starts'], entry['families'])
                    if family > 256}
        self.assertEqual(residues, set(reader_b.DESIGN_RESIDUES) | {240})      # 255 is 240 in the last family
        self.assertEqual(len(reader_b.replay_plan(reader_b.R2_NAMED, (0,), 1, reader_b.CAPACITY)), 2)

    def test_r4_runs_at_the_first_assignment_and_at_the_one_holding_c(self):
        """Idle segments must share a trace with a live segment at C (the served layout near 131k) too."""
        families = reader_b.family_plan(reader_b.CAPACITY, reader_b.R2_NAMED, reader_b.R2_FAMILIES)
        plan = reader_b.replay_plan(families, reader_b.DESIGN_RESIDUES, 2, reader_b.CAPACITY)
        self.assertNotIn(reader_b.CAPACITY, plan[0]['families'])
        self.assertEqual(reader_b.idle_entries(plan, reader_b.CAPACITY), [0, 2])
        self.assertEqual((plan[2]['families'][0], plan[2]['tables']), (reader_b.CAPACITY, True))
        watcher = reader_b.replay_plan(reader_b.R2_NAMED, (0, 32, 255), 1, reader_b.CAPACITY)
        self.assertEqual(reader_b.idle_entries(watcher, reader_b.CAPACITY), [0, 1])
        self.assertEqual(reader_b.idle_entries(reader_b.replay_plan([256, 512, 1280], (7,), 1, 1280), 1280), [0])

    def test_idle_patterns(self):
        self.assertEqual(reader_b.parse_patterns('3,2+3,0,0+1'), [(3,), (2, 3), (0,), (0, 1)])
        self.assertEqual(reader_b.parse_patterns('3+1'), [(1, 3)])
        self.assertEqual(reader_b.idle_assignment((3, 1)), {1: 0, 3: 32})
        self.assertEqual(reader_b.idle_assignment((2, 3)), {2: 0, 3: 32})
        self.assertEqual(reader_b.idle_assignment((0,)), {0: 0})
        for text in ('4', '1+1', '0+1+2', '0+1+2+3', 'x'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                reader_b.parse_patterns(text)

    def test_the_words_this_harness_expects(self):
        self.assertEqual(reader_b.expected_word(131312), [240, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(reader_b.expected_cur_pos(131312), [131327, 131327])
        self.assertEqual(reader_b.expected_cur_pos(32), [255, 255])
        for start in (0, 32, 128, 255, 256, 4351, 131311):
            self.assertEqual((reader_b.expected_word(start)[0], reader_b.expected_cur_pos(start)[0]),
                             extent_module.extent_values(start))

    def comparisons(self, kinds, differing=0):
        return [dict(section=section, kind=kind, label=kind, differing=differing, decisive=True)
                for section, names in reader_b.SECTION_KINDS.items() for kind in names if kind in kinds]

    def full_report(self):
        return dict(capacity=reader_b.CAPACITY, sections=list(reader_b.SECTIONS),
                    comparisons=self.comparisons(reader_b.DECISIVE_KINDS), liveness=[dict(label='l', live=True)],
                    r1_run={name: list(reader_b.R1_WORDS) for name in reader_b.R1_GEOMETRIES},
                    r2_families_replayed=reader_b.family_plan(reader_b.CAPACITY, reader_b.R2_NAMED, 56),
                    variants_run=['normal', 'peaky'], idle_starts_run=[0, 32], seeds_run=[0, 1, 2], failures=[])

    def test_the_decision_and_the_scope(self):
        report = self.full_report()
        decision = reader_b.decide(report)
        self.assertEqual((decision['verdict'], decision['scope'], decision['scope_short']), ('PASS', 'full', []))
        report['r2_families_replayed'] = report['r2_families_replayed'][:50]
        self.assertEqual(reader_b.decide(report)['scope_short'], ['r2_families'])
        report['variants_run'] = ['peaky']
        report['r1_run'].pop('G4B1')
        self.assertEqual(reader_b.decide(report)['scope_short'], ['r1', 'r2_families', 'variants'])
        report = self.full_report()
        report['comparisons'][3]['differing'] = 5
        self.assertEqual(reader_b.decide(report)['verdict'], 'FAIL')
        report['liveness'].append(dict(label='dead', live=False))
        self.assertEqual(reader_b.decide(report)['verdict'], 'NO-DECISION')
        report = self.full_report()
        report['comparisons'] = [entry for entry in report['comparisons'] if entry['section'] != 'R4']
        decision = reader_b.decide(report)
        self.assertEqual(decision['verdict'], 'NO-DECISION')
        self.assertIn('no decisive comparison of R4 ran', decision['reasons'])
        report = self.full_report()
        report['deadline'] = dict(seconds=5, skipped=['block/seed1'])
        self.assertEqual(reader_b.decide(report)['verdict'], 'NO-DECISION')
        report = self.full_report()
        report['failures'] = ['R1/seed0: boom']
        self.assertEqual(reader_b.decide(report)['verdict'], 'NO-DECISION')

    def test_a_full_scope_needs_the_block_run_of_seeds_0_1_and_2(self):
        """One seed over every section, family and variant is a reduced scope, never CB2b's evidence."""
        report = self.full_report()
        report['seeds_run'] = [0]
        decision = reader_b.decide(report)
        self.assertEqual((decision['verdict'], decision['scope'], decision['scope_short']),
                         ('PASS', 'reduced', ['seeds']))
        del report['seeds_run']
        self.assertEqual(reader_b.decide(report)['scope_short'], ['seeds'])
        report['seeds_run'] = [0, 1, 2, 3, 4]
        self.assertEqual(reader_b.decide(report)['scope'], 'full')

    def test_the_verdict_line(self):
        report = self.full_report()
        report['two_chip_view'] = dict(phantom_programs=12)
        report['modules'] = dict(sha256={'extent_attention_replay.py': 'ab' * 32})
        report['decision'] = reader_b.decide(report)
        line = reader_b.verdict_line(report)
        self.assertTrue(line.startswith('K64J_READER verdict=PASS scope=full r1=2/2 r1_reader=1/1 staging=6/6 r2=2/2 '
                                        'r2_trace=1/1 r4=3/3 live=1/1 families=56 chips=1of2 phantom=12 '
                                        'extent_sha256=' + 'ab' * 32), line)
        report['seeds_run'] = [0]
        report['decision'] = reader_b.decide(report)
        self.assertIn(' scope=reduced ', reader_b.verdict_line(report))
        self.assertIn(' scope_short=seeds', reader_b.verdict_line(report))

    def test_the_pindiag_lines_one_construction_must_log(self):
        engaged = '[PINDIAG] extent replay engaged segments=4 flags=0x27,0x27,0x27,0x27 mask=narrow capacity=4352'
        mode = ("[PINDIAG] sdpa qwen-modes modes=extent,share,slice,tail rows=16 capacity=4352 bundles=[2] "
                "flags=['0x27'] mask=narrow")
        binary = '[PINDIAG] sdpa qwen-modes binary /x carries the [QWEN-SDPA] branch with KV share'
        self.assertEqual(reader_b.pindiag_problems([binary] + [mode] * 4 + [engaged], 4352), [])
        self.assertEqual(len(reader_b.pindiag_problems([mode] * 4, 4352)), 1)
        self.assertEqual(len(reader_b.pindiag_problems([mode] * 3 + [engaged], 4352)), 1)
        self.assertEqual(len(reader_b.pindiag_problems([mode.replace('narrow', 'wide')] * 4 + [engaged], 4352)), 1)
        self.assertEqual(len(reader_b.pindiag_problems([mode] * 4 + [engaged.replace('0x27,0x27', '0x7,0x27')], 4352)),
                         1)

    def test_the_arguments(self):
        args = parse()
        self.assertEqual((args.capacity, args.sections, args.seeds, args.variants, args.r1_geometries, args.r1_words,
                          len(args.families), args.r2_restages, args.idle_patterns, args.ci_root, args.output_memory),
                         (131328, ['R1', 'S', 'R2', 'R4'], [0, 1, 2], ['normal', 'peaky'], ['G8B2', 'G4B3', 'G4B1'],
                          [0, 7, 32, 127, 128, 240, 255], 56, 2, [(3,), (2, 3), (0,), (0, 1)], '/bench/ci', 'l1'))
        self.assertEqual(args.served_root, '/experiment-scripts/ci')
        for bad in (['--sections', 'K2'], ['--variants', 'zeroq'], ['--r1-geometries', 'G8B3'], ['--r1-words', '256'],
                    ['--r2-restages', '3'], ['--r2-residues', ''], ['--idle-patterns', '0+1+2'], ['--capacity', '300'],
                    ['--expect-binary-sha256', 'abc'], ['--r2-families', '600']):
            with self.subTest(bad=bad), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                parse(bad)
        self.assertEqual(reader_b.section_runs(parse(['--seeds', '3,4'])),
                         [(3, 'R1'), (3, 'block'), (4, 'block')])
        self.assertEqual(reader_b.section_runs(parse(['--sections', 'R1'])), [(0, 'R1')])


# ---------------------------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    def test_the_pinned_shas_are_the_served_ones_and_the_runner_takes_them_from_the_image(self):
        import c2_overlay
        import test_extent_attention_replay as extent_tests
        served = extent_tests.StructureTests.SOURCES
        self.assertEqual(reader_b.PINNED, {name: served[name] for name in reader_b.PINNED})
        self.assertEqual(reader_b.PINNED['attention_mask_replay.py'], extent_tests.SERVED_MASK)
        self.assertEqual(sha(extent_tests.served_mask_source().encode('utf-8')), extent_tests.SERVED_MASK)
        # The frozen recipe adapts the mask module alone: the image holds the checkout's bytes of the other three.
        for name, digest in reader_b.PINNED.items():
            if name != 'attention_mask_replay.py':
                self.assertEqual(sha((CI / name).read_bytes()), digest, name)
        self.assertEqual(set(reader_b.SERVED_MODULES), {name[:-3] for name in reader_b.PINNED if name.endswith('.py')})
        self.assertEqual((reader_b.SERVED_ROOT, reader_b.CI_ROOT), (c2_overlay.PINNED_TREE, '/bench/ci'))
        runner = read(RUNNER)
        # The runner neither ships nor checks a checkout copy of a pinned source. The image's copies run, the harness
        # checks their bytes before it opens the device, and the container logs their sha256 first.
        self.assertEqual(re.findall(r'^(?:MASK|FOLD)_(?:PY|CPP)=', runner, flags=re.M), [])
        sources = re.search(r"^CI_SOURCES='([^']*)'$", runner, flags=re.M | re.S).group(1).split()
        self.assertEqual(sorted(sources), sorted(reader_b.RECORDED_SOURCES))
        self.assertEqual({name for _key, name in reader_b.MODULES} - set(reader_b.SERVED_MODULES),
                         {name[:-3] for name in sources})
        self.assertEqual(re.findall(r'^SERVED_CI=(.*)$', runner, flags=re.M), [reader_b.SERVED_ROOT])
        served_sources = re.search(r"^SERVED_SOURCES='([^']*)'$", runner, flags=re.M).group(1).split()
        self.assertEqual(sorted(served_sources), sorted(reader_b.PINNED))
        self.assertIn("exec python3 -B /bench/extent_reader_card_b.py \"$@\"", runner)

    def test_the_constants_are_the_served_geometrys(self):
        self.assertEqual(reader_b.SEGMENTS, tuple(zip(range(0, 64, 16), range(16, 80, 16))))
        layout = extent_module.LAYOUT(16, 8)
        self.assertEqual([[(group['offset'], group['rows']) for group in bundle] for bundle in layout],
                         [[(0, 8), (8, 8)]])
        self.assertEqual((reader_b.SERVED_FLAGS, reader_b.COMPILE_FLAGS, extent_module.EXTENT_FLAGS), (0x27, 0x7, 0x27))
        self.assertEqual(reader_b.ENGAGED_MARKER, extent_module.ENGAGED_MARKER)
        self.assertEqual(reader_b.MODES_MARKER, pooled_attention_replay.SDPA_MODES_MARKER)
        self.assertEqual(reader_b.IDLE_STARTS, (0, 32))
        self.assertEqual(reader_b.MIN_LIVE_START, extent_module.MIN_LIVE_START)

    def test_the_lent_storage_is_the_pools(self):
        """lend_storage allocates tensor for tensor what ServingBufferPool(extent_replay=True) lends for (4, 16)."""
        import test_extent_attention_replay as extent_tests
        from serving_buffer_pool import ServingBufferPool
        import serving_buffer_pool
        device = extent_tests.FakeDevice()
        helpers = [SimpleNamespace(allocate=lambda: [device.device_tensor((1, 1, 32), 'bf16', 'tile')])
                   for layer in range(48)]

        def rope(positions):
            return tuple(device.device_tensor((1, len(positions), 1, 64), 'bf16', 'tile') for side in (0, 1))

        with mock.patch('serving_buffer_pool.pindiag'):
            pool = ServingBufferPool(device, extent_tests.MESH, users=1, helpers=helpers, page_width=extent_tests.WIDTH,
                                     bucket_rows=(1,), rope=rope, packed_shapes=((4, 16),), packed_replay_group_rows=8,
                                     extent_replay=True)
        lent = pool.packed_extent(4, 16)
        mine = reader_b.lend_storage(device, extent_tests.MESH, torch, serving_buffer_pool, extent_tests.WIDTH)
        self.assertIsInstance(mine, serving_buffer_pool.PackedExtentStorage)

        def shape(storage):
            return [(tuple(tensor.shape), tensor.dtype, tensor.layout, tensor.memory, tensor.value.abs().sum().item())
                    for tensor in storage.tensors]

        self.assertEqual(shape(mine), shape(lent))
        self.assertEqual((mine.users, mine.rows), (lent.users, lent.rows))
        pool.close()

    def test_the_two_host_mirrors_agree(self):
        report = dict(failures=[])
        for name, (rows, batches, offset) in reader_b.R1_GEOMETRIES.items():
            self.assertTrue(reader_b.mirrors_agree(torch, extent_module, rows, batches, offset, range(256), report,
                                                   name))
        self.assertEqual(report['failures'], [])
        with mock.patch.object(extent_module, 'narrow_mask_host', lambda word, *a: card_b.served_mask(
                torch, word + 1, 256, rows=a[0], batches=a[1], offset=a[2])):
            self.assertFalse(reader_b.mirrors_agree(torch, extent_module, 8, 2, 0, [7], report, 'G8B2'))
        self.assertIn('disagree at word 7', report['failures'][0])

    def test_the_fake_kernels_are_the_mirrors(self):
        """The fake's emulations of the pinned kernels agree with the host mirrors (narrow_mask_host, fold_query,
        unfold_output), so the device flow below tests the reader, not the fake."""
        from attention_head_fold import fold_query, unfold_output
        fake = FakeReaderTtnn(torch)
        view = reader_b.TwoChipView(fake)
        for rows, batches, offset in reader_b.R1_GEOMETRIES.values():
            positions = extent_module._upload(view, fake, torch.zeros(8, dtype=torch.int32), view.int32)
            mask = extent_module._upload(view, fake, torch.zeros(batches, 1, rows * 12, 256, dtype=torch.bfloat16),
                                         view.bfloat16)
            with view.installed():
                program = extent_module.prepare_narrow(fake, positions, mask, rows=rows, batches=batches, offset=offset)
            for word in (0, 7, 32, 200, 255):
                fake.memory[positions.address] = torch.tensor([word] + [0] * 7, dtype=torch.int32)
                with view.installed():
                    extent_module.attention_mask_replay.execute(positions, mask, program)
                self.assertEqual(card.differing(torch, fake.memory[mask.address],
                                                extent_module.narrow_mask_host(word, rows, batches, offset)), 0)
        tokens = torch.randn(1, 16, 12, 256).to(torch.bfloat16)
        source = fake.from_torch(tokens, dtype=fake.bfloat16, layout=fake.TILE_LAYOUT, device=fake)
        owned = []
        from attention_fold_dma import device_layout_dma
        with view.installed():
            folded = device_layout_dma(fake, source, 8, owned, offset=8)
        self.assertTrue(torch.equal(fake.memory[folded.address], fold_query(tokens[:, 8:16])))
        with view.installed():
            back = device_layout_dma(fake, folded, 8, owned, inverse=True)
        self.assertTrue(torch.equal(fake.memory[back.address], unfold_output(fold_query(tokens[:, 8:16]), 8)))

    def test_the_two_chip_view(self):
        fake = FakeReaderTtnn(torch)
        view = reader_b.TwoChipView(fake)
        tensor = fake.from_torch(torch.zeros(8, dtype=torch.int32), dtype=fake.int32, layout=fake.ROW_MAJOR_LAYOUT,
                                 device=fake)
        shards = view.get_device_tensors(tensor)
        self.assertEqual([shard.buffer_address() for shard in shards], [tensor.address] * 2)
        self.assertIs(view.int32, fake.int32)
        program = view.MeshProgramDescriptor()
        for chip in (0, 1):
            coordinate = view.MeshCoordinate(0, chip)
            program[view.MeshCoordinateRange(coordinate, coordinate)] = 'chip%d' % chip
        with self.assertRaises(ValueError):
            program[view.MeshCoordinateRange((0, 2), (0, 2))] = 'chip2'
        with self.assertRaises(ValueError):
            program[view.MeshCoordinateRange((0, 1), (0, 1))] = 'again'
        real = program.realise()
        self.assertEqual(dict(real), {ONE: 'chip0'})
        self.assertIs(program.realise(), real)
        self.assertEqual((view.realised, view.phantom), (1, 1))
        with self.assertRaises(RuntimeError):
            program[view.MeshCoordinateRange((0, 0), (0, 0))] = 'late'
        half = view.MeshProgramDescriptor()
        half[view.MeshCoordinateRange((0, 0), (0, 0))] = 'chip0'
        with self.assertRaisesRegex(ValueError, 'every chip'):
            half.realise()
        two = mock.Mock(get_device_tensors=lambda value: [Shard(1), Shard(2)])
        with self.assertRaisesRegex(RuntimeError, 'ONE chip'):
            reader_b.TwoChipView(two).get_device_tensors(tensor)
        saved = sys.modules.get('ttnn')
        with view.installed():
            import ttnn
            self.assertIs(ttnn, view)
            with view.installed():
                self.assertIs(sys.modules['ttnn'], view)
            self.assertIs(sys.modules['ttnn'], view)
        self.assertIs(sys.modules.get('ttnn'), saved)

    def test_a_read_that_needs_the_shard_falls_back_to_it(self):
        shard = object()
        calls = []

        def to_torch(value):
            calls.append(value)
            if value is not shard:
                raise RuntimeError('a mesh composer is required')
            return torch.ones(2)

        ttnn = SimpleNamespace(to_torch=to_torch, get_device_tensors=lambda value: [shard])
        self.assertTrue(torch.equal(reader_b.read(ttnn, 'mesh tensor', 'x'), torch.ones(2)))
        self.assertEqual(calls, ['mesh tensor', shard])
        ttnn.get_device_tensors = lambda value: [shard, shard]
        with self.assertRaisesRegex(RuntimeError, 'mesh composer'):
            reader_b.read(ttnn, 'mesh tensor', 'x')

    def test_the_runners_cb2b_examples_name_card_m(self):
        """Card B is reserved for another project and there is no default QUAL_CARD: every CB2b example command in
        the runner's header names card M by its board id, with ALLOW_SERVING_CARD=1, and none names card B."""
        text = read(RUNNER)
        start = text.index('# CB2b (')
        paragraph = text[start:text.index(NL + '#' + NL, start)]
        commands = [line for line in paragraph.splitlines() if 'bash run_card_b.sh' in line]
        self.assertEqual(len(commands), 2, paragraph)
        for line in commands:
            self.assertIn('#   QUAL_CARD=%s ALLOW_SERVING_CARD=1 K64J_HARNESS=extent_reader ' % CARD_M, line)
        self.assertNotIn(CARD_B, paragraph)
        self.assertIn('C2_CARDM_ENV=K64J_HARNESS=extent_reader [WATCHER=1]', paragraph)

    def test_without_the_view_the_real_reader_refuses_one_chip(self):
        fake = FakeReaderTtnn(torch)
        positions = extent_module._upload(fake, fake, torch.zeros(8, dtype=torch.int32), fake.int32)
        mask = extent_module._upload(fake, fake, torch.zeros(2, 1, 96, 256, dtype=torch.bfloat16), fake.bfloat16)
        with mock.patch.dict(sys.modules, {'ttnn': fake}):
            with self.assertRaisesRegex(ValueError, 'Two chip-local metadata buffers required'):
                extent_module.prepare_narrow(fake, positions, mask, rows=8, batches=2, offset=0)


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------

def make_tree(directory):
    """This runner and what it ships, laid out as in the checkout under a temporary root: (runner, scripts/ci)."""
    root = Path(directory) / 'checkout'
    ops = root / 'optimisation' / 'ttnn-op'
    for source in (RUNNER, HERE / 'k64j_card_b.py', HERE / 'extent_reader_card_b.py',
                   PROBE_DIR / 'probe_k64j_card_b.py', PROBE_DIR / 'split_model.py',
                   OPS / 'sdpa_decode_qwen' / 'test_sdpa_decode_qwen_card_m.py',
                   OPS / 'sdpa_decode_qwen' / 'probe_k1_card_b.py'):
        (ops / source.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, ops / source.parent.name / source.name)
    ci = root / 'scripts' / 'ci'
    ci.mkdir(parents=True)
    for name in tuple(reader_b.RECORDED_SOURCES) + tuple(reader_b.QUAD_RECORDED_SOURCES):   # the pinned sources are the
        if not (ci / name).exists():                                                          # image's, never these
            shutil.copyfile(CI / name, ci / name)
    return ops / 'k64j' / 'run_card_b.sh', ci


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.graft = card_tests.make_graft(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, runner=RUNNER, **env):
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ['QUAL_CARD'] = CARD_X   # QUAL_CARD has no default
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='1',
                       K64J_HARNESS='extent_reader', KOPGRAFT64=self.graft.as_posix(),
                       EXPECT_TTNNCPP_SHA256=sha(card_tests.BINARY))
        environ.update(env)
        return subprocess.run([BASH, Path(runner).as_posix()], env=environ, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=120)

    def argv(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(lines), 1, result.stdout)
        return shlex.split(lines[0][len('### argv: '):])

    def mounts(self, argv):
        out = []
        for index, word in enumerate(argv):
            if word == '--mount':
                out.append(dict(part.split('=', 1) if '=' in part else (part, True)
                                for part in argv[index + 1].split(',')))
        return out

    def harness_args(self, argv):
        return reader_b.parse_args(argv[argv.index('card') + 1:])

    def test_the_extent_readers_dry_run(self):
        result = self.run_runner()
        argv = self.argv(result)
        self.assertEqual(result.stderr, '')
        self.assertIn('### code under test: ', result.stdout)
        self.assertIn(' harness=extent_reader', result.stdout)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-reader-' + CARD_X)
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_X)
        bench = {m['dst']: m for m in self.mounts(argv) if m['dst'].startswith('/bench/')}
        self.assertEqual(sorted(bench), ['/bench/ci', '/bench/extent_reader_card_b.py', '/bench/k64j_card_b.py',
                                         '/bench/probe_k1_card_b.py', '/bench/probe_k64j_card_b.py',
                                         '/bench/split_model.py', '/bench/test_sdpa_decode_qwen_card_m.py'])
        self.assertTrue(all(m.get('readonly') for m in bench.values()))
        self.assertTrue(bench['/bench/ci']['src'].endswith('/scripts/ci'), bench['/bench/ci'])
        self.assertTrue(bench['/bench/extent_reader_card_b.py']['src'].endswith('/k64j/extent_reader_card_b.py'))
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        for value in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', 'QWEN_FAST_SDPA_MODES=tail,share,slice',
                      'TT_METAL_CACHE=/kcache'):
            self.assertIn(value, env)
        self.assertNotIn('TT_METAL_WATCHER=5', env)
        inner = argv[argv.index('--entrypoint') + 4]
        self.assertTrue(inner.endswith('exec python3 -B /bench/extent_reader_card_b.py "$@"'), inner)
        for name in reader_b.PINNED:                            # the image's copies, logged before the harness runs
            self.assertIn('/experiment-scripts/ci/%s ' % name, inner.split('2>&1; ')[0])
        self.assertNotIn('/experiment-scripts/ci', ' '.join(m.get('src', '') for m in self.mounts(argv)))
        args = self.harness_args(argv)
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.deadline_s, args.ci_root, args.served_root,
                          args.sections, len(args.families)),
                         (sha(card_tests.BINARY), 300.0, 4800.0, '/bench/ci', '/experiment-scripts/ci',
                          ['R1', 'S', 'R2', 'R4'], 56))
        self.assertRegex(args.out.as_posix(), r'^/results/reader-[0-9]{8}T[0-9]{6}\.json$')
        # The default harness is untouched by the selection: no /bench/ci, no served sources, no QWEN_FAST_SDPA_MODES.
        argv = self.argv(self.run_runner(K64J_HARNESS='card'))
        self.assertNotIn('/bench/ci', [m['dst'] for m in self.mounts(argv)])
        self.assertNotIn('/experiment-scripts/ci', argv[argv.index('--entrypoint') + 4])
        self.assertNotIn('QWEN_FAST_SDPA_MODES=tail,share,slice', argv)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-card-' + CARD_X)

    def test_the_watcher_pass_and_the_full_pass(self):
        argv = self.argv(self.run_runner(WATCHER='1'))
        self.assertIn('TT_METAL_WATCHER=5', [argv[i + 1] for i, word in enumerate(argv) if word == '-e'])
        args = self.harness_args(argv)
        self.assertEqual((args.seeds, args.variants, args.r1_words, args.families, args.r2_restages,
                          args.idle_patterns, args.watchdog, args.deadline_s),
                         ([0], ['peaky'], [0, 32, 255], [256, 2304, 16640, 65792, 131328], 1, [(2, 3)], 120.0,
                          2100.0))
        argv = self.argv(self.run_runner(CARD_B_ARGS='--sections R1,S,R2,R4 --seeds 0,1,2 --variants normal,peaky'))
        args = self.harness_args(argv)
        self.assertEqual((args.seeds, args.variants, len(args.families), args.r2_restages, len(args.idle_patterns)),
                         ([0, 1, 2], ['normal', 'peaky'], 56, 2, 4))

    def test_an_unknown_harness_is_refused(self):
        result = self.run_runner(K64J_HARNESS='k2')
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: K64J_HARNESS=k2 is none of card', result.stderr)
        self.assertNotIn('### argv: ', result.stdout)

    def tree(self):
        return make_tree(self.dir)

    def test_a_missing_source_is_refused_before_launch_and_no_checkout_copy_of_a_pinned_one_is_needed(self):
        runner, ci = self.tree()
        self.assertFalse([name for name in reader_b.PINNED if (ci / name).exists()])
        argv = self.argv(self.run_runner(runner))               # the image's pinned sources run, never the checkout's
        inner = argv[argv.index('--entrypoint') + 4]
        self.assertTrue(all('/experiment-scripts/ci/' + name in inner for name in reader_b.PINNED), inner)
        (ci / 'serving_buffer_pool.py').unlink()
        result = self.run_runner(runner)
        self.assertEqual(result.returncode, 1)
        self.assertIn('serving_buffer_pool.py missing (the extent reader runs this checkout\'s scripts/ci)',
                      result.stderr)
        (runner.parent / 'extent_reader_card_b.py').unlink()
        shutil.copyfile(CI / 'serving_buffer_pool.py', ci / 'serving_buffer_pool.py')
        self.assertIn('extent_reader_card_b.py missing', self.run_runner(runner).stderr)
        self.argv(self.run_runner(runner, K64J_HARNESS='card'))          # the default harness needs neither

    def test_card_m_by_its_board_id_and_the_override(self):
        result = self.run_runner(QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1', WATCHER='1')
        argv = self.argv(result)
        self.assertIn('WARNING: ALLOW_SERVING_CARD=1: this run is on %s, card M' % CARD_M, result.stderr)
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_M)
        self.assertEqual((argv.count('--device'), argv[argv.index('--name') + 1]), (1, 'qwen-k64j-reader-card-m'))
        self.assertEqual(self.run_runner(QUAL_CARD=CARD_M).returncode, 1)

    def test_the_cardm_job_file_takes_the_documented_values(self):
        import c2_serving_job as job
        base = dict(C2_CARDM_HARNESS='optimisation/ttnn-op/k64j/run_card_b.sh')
        watcher = dict(base, C2_CARDM_ARGS='--sections R1,S,R2,R4',
                       C2_CARDM_ENV='K64J_HARNESS=extent_reader WATCHER=1 KOPGRAFT64=/home/thatch/opgraft-K64j '
                                    'EXPECT_TTNNCPP_SHA256=' + 'ab' * 32)
        full = dict(base, C2_CARDM_ARGS='--sections R1,S,R2,R4 --seeds 0,1,2 --variants normal,peaky',
                    C2_CARDM_ENV='K64J_HARNESS=extent_reader KOPGRAFT64=/home/thatch/opgraft-K64j '
                                 'EXPECT_TTNNCPP_SHA256=' + 'ab' * 32)
        for values in (watcher, full):
            harness, words, env = job.read_cardm(values, True, root=str(ROOT))
            self.assertEqual(harness, base['C2_CARDM_HARNESS'])
            self.assertIn('K64J_HARNESS=extent_reader', env.split())
            reader_b.parse_args(['--out', 'x.json'] + words.split())
        # The runner's docs name these keys.
        runner = read(RUNNER)
        self.assertIn('C2_CARDM_ENV=K64J_HARNESS=extent_reader [WATCHER=1] KOPGRAFT64=/home/thatch/opgraft-K64j',
                      runner)


@unittest.skipUnless(BASH, 'bash not found')
class CardMRunTests(unittest.TestCase):
    """The runner's REAL launch path for K64J_HARNESS=extent_reader on card M (card B is reserved), on
    test_qual_card's fake rig with docker, fuser, sudo, id and timeout stubbed, as test_k64j_card_b.CardMRunTests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rig = card_tests.qual_tests.FakeRig(self.dir / 'rig')
        self.graft = card_tests.make_graft(self.dir)
        self.runner, self.ci = make_tree(self.dir)
        self.launched = self.rig.dir / 'docker-run.argv'

    def tearDown(self):
        self.tmp.cleanup()

    def run_on(self, **env):
        text = read(self.runner)
        end = text.index(NL, text.index('# <<< qual_card.sh')) + 1
        stubs = self.rig.stubs() + [
            'docker() { case $1 in ps) echo %s ;; inspect) cat "$FAKE_DIR/container-$2" ;; image) return 0 ;; '
            'run) printf "%%q " "$@" > "$FAKE_DIR/docker-run.argv"; echo "K64J_READER verdict=PASS scope=full"; '
            'return "${FAKE_RUN_STATUS:-0}" ;; rm) return 0 ;; esac; }' % ' '.join(self.rig.containers),
            'fuser() { local n held=1; for n in "$@"; do case " ${FAKE_HELD:-} " in *" $n "*) '
            'echo "$n: thatch 4242 F.... python3" >&2; held=0 ;; esac; done; return $held; }',
            'sudo() { return 1; }',
            'id() { echo 1000; }',
            'timeout() { while [ $# -gt 0 ]; do case $1 in -k) shift 2 ;; [0-9]*) shift; break ;; *) break ;; esac; '
            'done; "$@"; }',
        ]
        self.runner.write_bytes((text[:end] + NL.join(stubs) + NL + text[end:]).encode('utf-8'))
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='0',
                       KOPGRAFT64=self.graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(card_tests.BINARY),
                       QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1', K64J_HARNESS='extent_reader')
        environ.update(env)
        return subprocess.run([BASH, self.runner.as_posix()], env=environ, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=120)

    def test_card_m_launches_the_extent_reader_on_its_node(self):
        node = self.rig.node(CARD_M)
        result = self.run_on(WATCHER='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('### K64J_READER verdict=PASS scope=full', result.stdout)
        argv = shlex.split(self.launched.read_text(encoding='utf-8'))
        self.assertEqual((argv.count('--device'), argv[argv.index('--device') + 1]), (1, node))
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-reader-card-m')
        mounts = [argv[index + 1] for index, word in enumerate(argv) if word == '--mount']
        ci = [mount for mount in mounts if ',dst=/bench/ci,' in mount]
        self.assertEqual(len(ci), 1, mounts)
        self.assertTrue(ci[0].endswith('/checkout/scripts/ci,dst=/bench/ci,readonly'), ci)
        args = reader_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.watchdog, args.deadline_s, args.seeds), (120.0, 2100.0, [0]))
        self.assertTrue(list((self.dir / 'results').glob('reader-*.log')))

    def test_a_host_holder_of_card_m_refuses(self):
        node = self.rig.node(CARD_M)
        result = self.run_on(FAKE_HELD=node)
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: host processes hold %s %s' % (node, self.rig.node(CARD_A)), result.stderr)
        self.assertFalse(self.launched.exists())


class ServedLoaderTests(unittest.TestCase):
    """load_modules in a fresh interpreter, as the container runs it. The pinned modules come from a served tree and
    the code under test from the checkout's scripts/ci. The served tree is rebuilt from git: the frozen recipe's stage
    of attention_mask_replay.py, and the checkout's bytes of the other three, which the image holds unchanged."""

    def setUp(self):
        import test_extent_attention_replay as extent_tests
        self.tmp = tempfile.TemporaryDirectory()
        self.served = Path(self.tmp.name) / 'experiment-scripts' / 'ci'
        self.served.mkdir(parents=True)
        with open(self.served / 'attention_mask_replay.py', 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(extent_tests.served_mask_source())
        for name in ('attention_mask_replay.cpp', 'attention_fold_dma.py', 'attention_fold_dma.cpp',
                     'frozen_context_geometry.py'):
            shutil.copyfile(CI / name, self.served / name)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, prelude=''):
        script = NL.join((
            'import json, sys',
            'sys.path[:0] = %r' % [str(HERE), str(PROBE_DIR), str(OPS / 'sdpa_decode_qwen')],
            prelude,
            'import extent_reader_card_b as reader_b',
            'report = dict(failures=[])',
            'mods = reader_b.load_modules(%r, report, %r)' % (str(CI), str(self.served)),
            'out = dict(failures=report["failures"], modules=report.get("modules"), loaded=mods is not None)',
            'if mods is not None:',
            '    out.update(bound=[mods.extent.attention_mask_replay is mods.mask,',
            '                      mods.extent.device_layout_dma is mods.fold.device_layout_dma,',
            '                      mods.pooled.validate_ticket is mods.mask.validate_ticket,',
            '                      sys.modules["attention_replay"].prepare is mods.mask.prepare,',
            '                      sys.modules["attention_parallel"].device_layout_dma',
            '                      is mods.fold.device_layout_dma])',
            '    out.update(geometry=sys.modules["frozen_context_geometry"].__file__,',
            '               served_on_path=%r in sys.path)' % str(self.served.resolve()),
            'print("LOADED " + json.dumps(out))'))
        environ = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=300, env=environ)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith('LOADED ')]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return json.loads(lines[0][len('LOADED '):])

    def test_the_pinned_modules_are_the_served_tree_s_and_the_code_under_test_binds_to_them(self):
        out = self.load()
        self.assertEqual((out['failures'], out['loaded']), ([], True))
        self.assertEqual(out['bound'], [True] * 5)
        files = {name: Path(path) for name, path in out['modules']['files'].items()}
        for name in reader_b.SERVED_MODULES:
            self.assertEqual(files[name], (self.served / (name + '.py')).resolve(), name)
        for _key, name in reader_b.MODULES:
            if name not in reader_b.SERVED_MODULES:
                self.assertEqual(files[name].parent, CI.resolve(), name)
        self.assertEqual({name: out['modules']['sha256'][name] for name in reader_b.PINNED}, reader_b.PINNED)
        self.assertEqual(out['modules']['sha256']['extent_attention_replay.py'],
                         sha((CI / 'extent_attention_replay.py').read_bytes()))
        self.assertEqual(Path(out['modules']['served_root']), self.served.resolve())
        # The served mask module's own import resolved in the served tree, as in serving, and the served tree left
        # sys.path afterwards (the code under test resolves in the checkout's).
        self.assertEqual(Path(out['geometry']).resolve(), (self.served / 'frozen_context_geometry.py').resolve())
        self.assertFalse(out['served_on_path'])

    def test_a_served_tree_without_the_served_bytes_decides_nothing(self):
        with open(self.served / 'attention_mask_replay.cpp', 'ab') as handle:
            handle.write(b'// drift\n')
        out = self.load()
        self.assertFalse(out['loaded'])
        self.assertTrue(any(failure.startswith('pinned source attention_mask_replay.cpp is ')
                            for failure in out['failures']), out['failures'])
        # The checkout's mask module where the image's belongs: the other version, refused.
        shutil.copyfile(CI / 'attention_mask_replay.cpp', self.served / 'attention_mask_replay.cpp')
        shutil.copyfile(CI / 'attention_mask_replay.py', self.served / 'attention_mask_replay.py')
        out = self.load()
        self.assertFalse(out['loaded'])
        checkout = sha((CI / 'attention_mask_replay.py').read_bytes())
        self.assertIn('pinned source attention_mask_replay.py is %s, not its served %s'
                      % (checkout[:16], reader_b.PINNED['attention_mask_replay.py'][:16]), out['failures'])

    def test_a_missing_served_module_or_one_imported_first_decides_nothing(self):
        out = self.load(prelude='sys.path.insert(0, %r); import attention_mask_replay' % str(CI))
        self.assertFalse(out['loaded'])
        self.assertTrue(any(failure.startswith('attention_mask_replay was imported from ')
                            and '--served-root' in failure for failure in out['failures']), out['failures'])
        (self.served / 'attention_fold_dma.py').unlink()
        out = self.load()
        self.assertFalse(out['loaded'])
        self.assertTrue(any(failure.startswith('attention_fold_dma.py is not in --served-root ')
                            for failure in out['failures']), out['failures'])


# ---------------------------------------------------------------------------------------------
# The device flow on the fake one-chip ttnn, through the real reader classes.
# ---------------------------------------------------------------------------------------------

class FlowBase(unittest.TestCase):
    """The harness end to end on a fake one-chip ttnn: the fixture and the runner every flow test uses. A width's class
    sets the harness's own --width arguments, the launch variables it needs and the patches of its geometry."""

    FULL = ['--capacity', '2304', '--r2-named', '256,512,2304', '--r2-families', '6', '--r2-residues',
            '0,7,240,255', '--seeds', '0', '--variants', 'normal,peaky', '--r1-words', '0,7,32,240,255',
            '--idle-patterns', '3,2+3,0']
    SMALL = ['--capacity', '1280', '--r2-named', '256,512,1280', '--r2-families', '5', '--r2-residues', '7,250',
             '--seeds', '0', '--variants', 'normal', '--r1-words', '7,255', '--r1-geometries', 'G8B2',
             '--r2-restages', '1', '--idle-patterns', '2+3']
    WIDTH_ARGV = ()
    WIDTH_ENV = {}

    def width_patches(self):
        return ()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        graft = card_tests.make_graft(self.dir)
        self.kernels = graft / 'sdpa_decode' / 'device' / 'kernels'
        self.binary = graft / '_ttnncpp.so'

    def tearDown(self):
        self.tmp.cleanup()

    def run_reader(self, fake, extra=(), base=None, env=None, patches=(), name='reader'):
        out = self.dir / ('%s.json' % name)
        fake.report_path = out
        markers = dict(flags=True, share=True, stage1=False)
        argv = ['--out', str(out), '--kernel-root', str(self.kernels), '--ci-root', '', '--served-root', '',
                '--expect-binary-sha256', sha(self.binary.read_bytes())] + list(self.WIDTH_ARGV)
        environ = {card.SCRATCH_ENV: '1', 'QWEN_FAST_SDPA_MODES': 'tail,share,slice'}
        environ.update(self.WIDTH_ENV)
        environ.update(env or {})
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {'ttnn': fake}))
            stack.enter_context(mock.patch.object(card, 'loaded_binary', return_value=(str(self.binary), markers)))
            stack.enter_context(mock.patch.dict(os.environ, environ))
            stack.enter_context(mock.patch.object(probe, 'clock', fake.clock))
            stack.enter_context(mock.patch.object(card, 'WATCHDOG', card.WATCHDOG))
            stack.enter_context(mock.patch.object(probe, 'WATCHDOG', probe.WATCHDOG))
            stack.enter_context(mock.patch.object(probe.k1, 'WATCHDOG', probe.k1.WATCHDOG))
            stack.enter_context(mock.patch.object(probe, 'DEADLINE', probe.DEADLINE))
            stack.enter_context(mock.patch.object(probe, 'print', create=True))
            stack.enter_context(mock.patch.object(reader_b, 'print', create=True))
            stack.enter_context(mock.patch.object(card_b, 'print', create=True))
            stack.enter_context(mock.patch.object(extent_module, 'print', create=True))          # _pindiag's lines
            stack.enter_context(mock.patch.object(pooled_attention_replay, 'print', create=True))
            stack.enter_context(mock.patch.object(sys, 'path', list(sys.path)))
            # In process, the pinned modules are this checkout's (--served-root ''). ServedLoaderTests covers the
            # served tree.
            stack.enter_context(mock.patch.object(reader_b, 'PINNED', checkout_pinned()))
            stack.enter_context(mock.patch('sys.stdout'))
            stack.enter_context(mock.patch('pooled_attention_replay._binary_checked', []))
            stack.enter_context(mock.patch('pooled_attention_replay.loaded_binary_has_modes',
                                           return_value=('/k64j/_ttnncpp.so', True)))
            for patch in list(self.width_patches()) + list(patches):
                stack.enter_context(patch)
            status = reader_b.main(argv + list(self.FULL if base is None else base) + list(extra))
        return status, json.loads(out.read_text())

    def kinds(self, report):
        return {kind: (row['equal'], row['runs']) for kind, row in report['tally'].items()}

    def failing(self, report):
        return {kind for kind, (equal, runs) in self.kinds(report).items() if equal != runs}


class DryRunTests(FlowBase):
    """The harness end to end on the fake: a 2,304-key table (9 chunks), the families 256 / 512 / 2,304 and three
    spread ones. One FULL run covers every section; each broken variant runs only what must catch it, on a 1,280-key
    table whose five families make two assignments (SMALL)."""

    # C = 1,280 again, but the named families put C in the second assignment: [256, 512, 768, 1024], then
    # [1280, 256, 512, 768] - segment 0 live at C while segments 2 and 3 go idle.
    AT_C = ['--capacity', '1280', '--r2-named', '256,512,768,1024,1280', '--r2-families', '5', '--r2-residues',
            '7,250', '--seeds', '0', '--variants', 'normal', '--r1-words', '7,255', '--r1-geometries', 'G8B2',
            '--r2-restages', '1', '--idle-patterns', '2+3']

    def test_pass_end_to_end(self):
        fake = FakeReaderTtnn(torch)
        status, report = self.run_reader(fake)
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict']), (0, True, 'PASS'))
        self.assertEqual(report['sections_done'], ['R1/seed0', 'block/seed0'])
        kinds = self.kinds(report)
        self.assertEqual(kinds['r1_eager'], (15, 15))                   # 3 geometries x 5 words
        self.assertEqual(kinds['r1_trace'], (15, 15))
        self.assertEqual(kinds['r1_reader'], (32, 32))                  # 8 replays x 4 segments
        self.assertEqual(kinds['staging_word'], (4, 4))
        self.assertEqual(kinds['staging_cur_pos'], (4, 4))
        self.assertEqual(kinds['staging_table'], (4, 4))
        self.assertEqual(kinds['restage_word'], (40, 40))               # 8 replays x 4 + 2 x 4 idle segments
        self.assertEqual(kinds['restage_table'], (24, 24))              # 4 table restages x 4 + 2 x 4 idle
        self.assertEqual(kinds['r2_construction_vs_wide'], (4, 4))
        self.assertEqual(kinds['r2_trace_vs_wide'], (32, 32))
        self.assertEqual(kinds['r2_trace_vs_eager'], (8, 8))
        self.assertEqual(kinds['r4_live_unchanged'], (16, 16))          # (3 + 2 + 3) x 2 variants
        self.assertEqual(kinds['r4_idle_finite'], (8, 8))
        self.assertEqual(kinds['r4_idle_vs_wide'], (8, 8))
        self.assertEqual(len(report['liveness']), 4)
        self.assertTrue(all(entry['live'] for entry in report['liveness']), report['liveness'])
        self.assertEqual(report['r2_families_replayed'], [256, 512, 768, 1280, 2048, 2304])
        self.assertEqual((report['variants_run'], report['idle_starts_run'], report['captures'], report['seeds_run']),
                         (['normal', 'peaky'], [0, 32], 1, [0]))
        self.assertIn('seeds', report['decision']['scope_short'])
        self.assertIn(' extent_sha256=%s' % sha((CI / 'extent_attention_replay.py').read_bytes()),
                      report['verdict_line'])
        self.assertEqual(report['extent_reader'], dict(flags=['0x27'] * 4, segments=4, capacity=2304, borrowed=8))
        self.assertIn([0x27, 2, 2304 // 32, 8], report['requested_programs'])
        for family in (256, 512, 768, 1280, 2048, 2304):
            self.assertIn([0x7, 2, family // 32, family // 32], report['requested_programs'])
        self.assertEqual([(line['entries'], line['kv_share'], line['q_slice']) for line in report['extent_lines']],
                         [(2, 'true', 'true')])
        self.assertFalse({'width', 'heads', 'kv_heads', 'chip_view', 'sibling_drift'} & set(report),
                         'the pair report keeps exactly its keys')
        self.assertEqual((report['served']['flags'], report['served']['reference_flags']), ('0x27', '0x7'))
        view = report['two_chip_view']
        self.assertEqual((view['chips_physical'], view['chips_presented']), (1, 2))
        self.assertEqual(view['phantom_programs'], view['programs_realised'])
        self.assertGreater(view['launches'], 100)
        self.assertEqual(report['modules']['sha256']['attention_mask_replay.cpp'],
                         reader_b.PINNED['attention_mask_replay.cpp'])
        self.assertEqual(report['modules']['sha256']['extent_attention_replay.py'],
                         sha((CI / 'extent_attention_replay.py').read_bytes()))
        self.assertEqual(len([line for line in report['pindiag'] if line.startswith(reader_b.ENGAGED_MARKER)]), 1)
        self.assertEqual(report['decision']['scope'], 'reduced')
        self.assertTrue(report['verdict_line'].startswith(
            'K64J_READER verdict=PASS scope=reduced r1=30/30 r1_reader=32/32 staging=116/116 r2=36/36 r2_trace=8/8 '
            'r4=32/32 live=4/4 families=6 chips=1of2'), report['verdict_line'])
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))
        self.assertIn(('attention_mask_replay.cpp', 48), fake.launched)
        self.assertIn(('attention_fold_dma.cpp', 64), fake.launched)

    def reader_patch(self, name, wrapper):
        original = getattr(extent_module.ExtentSegmentReader, name)
        return mock.patch.object(extent_module.ExtentSegmentReader, name, wrapper(original))

    def test_a_stale_cur_pos_fails(self):
        """cur_pos written at construction only: the restage checks and R2 catch it."""
        def wrapper(original):
            def stage_values(self, start, table):
                values = original(self, start, table)
                if self.start is None:
                    return values
                return [value for value in values if not any(value[0] is positions for positions in self.cur_pos)]
            return stage_values

        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, extra=['--sections', 'S,R2'],
                                         patches=[self.reader_patch('stage_values', wrapper)])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'restage_cur_pos', 'r2_trace_vs_wide'})

    def test_an_absolute_word_fails(self):
        """The word staged as the absolute start: S, the reader's masks (R1) and R2 catch it."""
        def absolute(start):
            return start, extent_module.extent(start) - 1

        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL,
                                         extra=['--sections', 'R1,S,R2'],
                                         patches=[mock.patch.object(extent_module, 'extent_values', absolute)])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertLessEqual({'staging_word', 'restage_word', 'r1_reader', 'r2_construction_vs_wide',
                              'r2_trace_vs_wide'}, self.failing(report))
        self.assertNotIn('r1_eager', self.failing(report))

    def test_a_wide_mask_read_fails_r2_only(self):
        """The device reads the narrow mask as a wide one ([C - 256, C): zeros): the mask and the staging are right,
        the attention is not."""
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'tail_at_capacity'}), base=self.SMALL,
                                         extra=['--sections', 'R1,S,R2'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        failing = self.failing(report)
        self.assertLessEqual({'r2_construction_vs_wide', 'r2_trace_vs_wide'}, failing)
        self.assertFalse(failing & {'r1_eager', 'r1_trace', 'r1_reader', 'staging_word', 'staging_cur_pos'})

    def missing_slot(self):
        def wrapper(original):
            def stage_values(self, start, table):
                values = original(self, start, table)
                out = []
                for destination, value, dtype, layout in values:
                    if any(destination is positions for positions in self.cur_pos):
                        value = value.clone()
                        value[1:] = 0
                    out.append((destination, value, dtype, layout))
                return out
            return stage_values
        return self.reader_patch('stage_values', wrapper)

    def test_a_missing_slot_copy_fails_the_staging(self):
        """cur_pos slot 1 never written: the staging readback catches it even where share hides it in the output."""
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, extra=['--sections', 'S,R2'],
                                         patches=[self.missing_slot()])
        self.assertEqual((status, report['decision']['verdict']), (1, 'FAIL'))
        self.assertEqual(self.failing(report), {'staging_cur_pos', 'restage_cur_pos'})
        # With a device that lacks R10's slot copy (each entry its own word) the attention moves too.
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'own_slot'}), base=self.SMALL,
                                         extra=['--sections', 'S,R2'], patches=[self.missing_slot()], name='own')
        self.assertLessEqual({'staging_cur_pos', 'r2_construction_vs_wide', 'r2_trace_vs_wide'}, self.failing(report))

    def test_an_unstaged_construction_fails(self):
        """Construction that stages nothing (the reader only records its start): S and the construction-state call
        catch it; the replays, restaged, do not."""
        def wrapper(original):
            def stage(self, start, table=None):
                if self.start is None:
                    self.start = start
                    return None
                return original(self, start, table)
            return stage

        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, extra=['--sections', 'S,R2'],
                                         patches=[self.reader_patch('stage', wrapper)])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'staging_word', 'staging_cur_pos', 'staging_table',
                                                'r2_construction_vs_wide'})

    def test_a_mask_kernel_that_skips_a_tile_fails(self):
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'mask_stale_tile'}), base=self.SMALL,
                                         extra=['--sections', 'R1,R2'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'FAIL'))
        self.assertLessEqual({'r1_eager', 'r1_trace', 'r1_reader', 'r2_trace_vs_wide'}, self.failing(report))

    def test_a_trace_that_keeps_its_captured_cur_pos_fails(self):
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'stale_trace'}), base=self.SMALL,
                                         extra=['--sections', 'R2'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'FAIL'))
        self.assertLessEqual({'r2_trace_vs_eager', 'r2_trace_vs_wide'}, self.failing(report))
        self.assertNotIn('r2_construction_vs_wide', self.failing(report))

    def test_a_nonfinite_idle_row_fails_r4(self):
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'idle_nonfinite'}), base=self.SMALL,
                                         extra=['--sections', 'R4'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'FAIL'))
        self.assertLessEqual({'r4_idle_finite'}, self.failing(report))
        self.assertNotIn('r4_live_unchanged', self.failing(report))

    def test_r1_trace_is_a_replay_not_a_launch(self):
        """A replayed mask launch on the word it was captured with (design Q1's in-trace half): only a real replay
        shows it - r1_trace fails at word 7 (the capture held 255, the last eager word), r1_eager passes."""
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'mask_trace_word'}), base=self.SMALL,
                                         extra=['--sections', 'R1'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'r1_trace'})
        self.assertEqual((self.kinds(report)['r1_trace'], self.kinds(report)['r1_eager']), ((1, 2), (2, 2)))
        self.assertEqual(report['decision']['first_differing'], ['R1/G8B2/word7/trace'])

    def test_r1_poisons_the_mask_before_every_run(self):
        """A replayed mask launch that skips a tile whose bits are the same at both words (column tile 0: no key of
        it is past row 240): what the eager runs left there is right, so only the NaN poison shows the tile was not
        rewritten."""
        for word in (240, 255):
            self.assertTrue(bool((extent_module.narrow_mask_host(word, 8, 2, 0)[..., :32] == 0).all()))
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'mask_trace_skips_tile'}), base=self.SMALL,
                                         extra=['--sections', 'R1', '--r1-words', '240,255'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'r1_trace'})
        self.assertEqual((self.kinds(report)['r1_trace'], self.kinds(report)['r1_eager']), ((0, 2), (2, 2)))

    def test_an_idle_segment_that_moves_its_live_neighbour_fails_r4_live_unchanged(self):
        """Segments 2 and 3 idle move segment 1's rows: the idle rows stay right, so only the live segments' check
        against the all-live replay can see it."""
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'idle_bleeds'}), base=self.SMALL,
                                         extra=['--sections', 'R4'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'r4_live_unchanged'})
        self.assertEqual(report['decision']['first_differing'], ['R2/seed0/normal/a0.0/idle2+3/segment1'])

    def test_r4_runs_again_at_the_assignment_holding_c(self):
        """R4 at the first assignment (no segment at C here) and again at the one holding C: an idle segment that
        disturbs a live segment at C is caught there, and only there."""
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.AT_C, extra=['--sections', 'R4'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (0, [], 'PASS'))
        labels = [entry['label'] for entry in report['comparisons'] if entry['kind'] == 'r4_live_unchanged']
        self.assertEqual(labels, ['R2/seed0/normal/a%d.0/idle2+3/segment%d' % (assignment, segment)
                                  for assignment in (0, 1) for segment in (0, 1)])
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'idle_bleeds_at_capacity'}), base=self.AT_C,
                                         extra=['--sections', 'R4'], name='bleeds')
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'r4_live_unchanged'})
        self.assertEqual(report['decision']['first_differing'], ['R2/seed0/normal/a1.0/idle2+3/segment0'])

    def test_a_device_that_finds_e_without_cur_pos_decides_nothing(self):
        """A 0x20 program that ignores cur_pos_tensor but takes the right E from the poisoned table: every R2 and S
        comparison is equal, so the cur_pos liveness control alone keeps this from a PASS."""
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'extent_from_table'}), base=self.SMALL,
                                         extra=['--sections', 'S,R2'])
        self.assertEqual((report['failures'], self.failing(report)), ([], set()))
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertEqual([entry['kind'] for entry in report['liveness'] if not entry['live']], ['cur_pos_live'])
        self.assertIn('1 liveness controls did not move', report['decision']['reasons'][0])

    def test_a_device_that_ignores_every_mask_decides_nothing(self):
        """Every call ignores its mask, the reader's and the 0x7 references' alike: R2 is equal everywhere, so the
        mask liveness control alone keeps this from a PASS."""
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'ignore_mask'}), base=self.SMALL,
                                         extra=['--sections', 'S,R2'])
        self.assertEqual((report['failures'], self.failing(report)), ([], set()))
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertEqual([entry['kind'] for entry in report['liveness'] if not entry['live']], ['mask_live'])

    def test_a_reader_that_never_logs_it_was_engaged_decides_nothing(self):
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, extra=['--sections', 'S'],
                                         patches=[mock.patch.object(extent_module, 'ENGAGED_MARKER', '[PINDIAG] x')])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertTrue(any('PINDIAG' in failure for failure in report['failures']), report['failures'])

    def test_a_binary_that_never_logs_f22_decides_nothing(self):
        status, report = self.run_reader(FakeReaderTtnn(torch, broken={'no_f22'}), base=self.SMALL,
                                         extra=['--sections', 'R2'])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any(failure.startswith('extent log: ') for failure in report['failures']), report['failures'])

    def test_the_code_under_test_must_come_from_the_ci_root_and_keep_the_pinned_bytes(self):
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL,
                                         extra=['--sections', 'S', '--ci-root', str(self.dir)])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any('not from --ci-root' in failure for failure in report['failures']), report['failures'])
        self.assertEqual(report['comparisons'], [])
        pinned = dict(checkout_pinned(), **{'attention_mask_replay.cpp': '0' * 64})
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, extra=['--sections', 'S'],
                                         patches=[mock.patch.object(reader_b, 'PINNED', pinned)])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any(failure.startswith('pinned source attention_mask_replay.cpp') for failure in
                            report['failures']), report['failures'])
        # A served tree the pinned modules were not loaded from decides nothing either (here they came first).
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL,
                                         extra=['--sections', 'S', '--served-root', str(self.dir)])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any('--served-root' in failure for failure in report['failures']), report['failures'])
        self.assertEqual(report['comparisons'], [])

    def test_the_sdpa_modes_and_the_scratch_are_checked_first(self):
        for env, needle in (({'QWEN_FAST_SDPA_MODES': 'tail,share'}, 'QWEN_FAST_SDPA_MODES must be'),
                            ({card.SCRATCH_ENV: '0'}, 'QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required')):
            with self.subTest(env=env):
                status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, env=env)
                self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
                self.assertTrue(any(needle in failure for failure in report['failures']), report['failures'])
                self.assertEqual(report['comparisons'], [])

    def test_the_deadline_stops_cleanly_and_lists_the_rest(self):
        fake = FakeReaderTtnn(torch, seconds_per_call=10.0)
        status, report = self.run_reader(fake, base=self.SMALL, extra=['--seeds', '0,1', '--deadline-s', '100'])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(report['deadline']['skipped'], report.get('deadline'))
        self.assertNotIn(1, report.get('seeds_run') or [])
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))


# ---------------------------------------------------------------------------------------------
# WIDTH 4: the four-card reader twin on one KV head (--width 4, run_card_b.sh TP4_WIDTH=4).
# ---------------------------------------------------------------------------------------------

QUAD = reader_b.QUAD
PAIR = reader_b.PAIR
TEMPLATES = CI / 'references' / 'tp4-s2-serve-jobs'


def four():
    """The launch variable that carries the four-card width into the process (tp_shapes reads it at call time)."""
    return mock.patch.dict(os.environ, {reader_b.TP_ENV: '4'})


def template(name):
    """{KEY: value} of a committed job template (comment lines and blanks skipped)."""
    values = {}
    for line in read(TEMPLATES / name).splitlines():
        if line.strip() and not line.startswith('#'):
            key, _, value = line.partition('=')
            values[key] = value
    return values


class FakeQuadTtnn(FakeReaderTtnn):
    """FakeReaderTtnn for the four-card chip: its mask and fold kernels are the _tp siblings, built for 6 head rows."""
    head_rows = 6


class QuadHelperTests(unittest.TestCase):
    full_report = HelperTests.full_report
    comparisons = HelperTests.comparisons

    def test_the_two_geometries_and_what_the_recorder_and_the_admission_expect_of_the_four_card_one(self):
        self.assertEqual((PAIR.width, PAIR.heads, PAIR.kv_heads, PAIR.group, PAIR.served_flags, PAIR.compile_flags,
                          PAIR.chips, PAIR.view_key, PAIR.reader_source),
                         (2, 12, 2, 6, 0x27, 0x7, 2, 'two_chip_view', 'extent_attention_replay.py'))
        self.assertEqual((reader_b.SERVED_FLAGS, reader_b.COMPILE_FLAGS, reader_b.MODULES[0]),
                         (0x27, 0x7, ('extent', 'extent_attention_replay')), 'the pair\'s constants never move')
        self.assertEqual((QUAD.width, QUAD.heads, QUAD.kv_heads, QUAD.group, QUAD.served_flags, QUAD.compile_flags,
                          QUAD.chips, QUAD.view_key, QUAD.reader_source),
                         (4, 6, 1, 6, 0x23, 0x3, 4, 'chip_view', 'extent_attention_replay_tp.py'))
        self.assertEqual(QUAD.served_flags, extent_tp.EXTENT_FLAGS, 'the twin serves exactly these flags')
        self.assertEqual(reader_b.GEOMETRIES, {2: PAIR, 4: QUAD})
        self.assertEqual('0x%x' % QUAD.served_flags, recorder.SERVED_FLAGS)
        self.assertEqual('0x%x' % QUAD.served_flags, admission.CB2B_SERVED_FLAGS_TP4)
        self.assertEqual(('1of%d' % QUAD.chips,), admission.CB2B_CHIPS_TP4)
        self.assertEqual(reader_b.QUAD_READER, recorder.READER_TP)
        self.assertEqual(reader_b.QUAD_SIBLINGS, recorder.PINNED_SIBLINGS)
        self.assertEqual(reader_b.CAPACITY, admission.CB2B_CAPACITY)
        self.assertEqual(tuple(reader_b.R1_GEOMETRIES), admission.CB2B_R1_GEOMETRIES)
        self.assertEqual(reader_b.DESIGN_RESIDUES, admission.CB2B_RESIDUES)
        self.assertEqual((reader_b.R2_NAMED, reader_b.R2_MIN_FAMILIES, reader_b.IDLE_STARTS),
                         (admission.CB2B_R2_NAMED, admission.CB2B_R2_MIN_FAMILIES, admission.CB2B_IDLE_STARTS))
        # Every source the four-card run records is one the runner requires, and the siblings are the image's.
        self.assertIn(reader_b.QUAD_READER, reader_b.QUAD_RECORDED_SOURCES)
        self.assertFalse(set(reader_b.QUAD_SIBLINGS) & set(reader_b.QUAD_RECORDED_SOURCES))
        self.assertEqual({name.rsplit('.', 1)[0] for name in reader_b.PINNED} | {'attention_fold_dma_tp'},
                         set(reader_b.QUAD_SERVED_MODULES))
        self.assertEqual({name for _key, name in reader_b.QUAD_MODULES if name in reader_b.QUAD_SERVED_MODULES},
                         set(reader_b.QUAD_SERVED_MODULES))

    def test_the_width_argument_changes_nothing_else(self):
        default, quad = parse(), parse(['--width', '4'])
        self.assertEqual((default.width, default.geometry, quad.width, quad.geometry), (2, PAIR, 4, QUAD))
        self.assertEqual({key for key in vars(default) if vars(default)[key] != vars(quad)[key]}, {'width', 'geometry'})
        for bad in (['--width', '3'], ['--width', '1'], ['--width', 'four']):
            with self.subTest(bad=bad), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                parse(bad)
        # The full pass at four cards is the harness's own defaults: what EV-F3 leaves to them.
        self.assertEqual((quad.capacity, quad.sections, quad.seeds, quad.variants, quad.r1_geometries, len(quad.families),
                          quad.r2_restages), (131328, ['R1', 'S', 'R2', 'R4'], [0, 1, 2], ['normal', 'peaky'],
                                              ['G8B2', 'G4B3', 'G4B1'], 56, 2))

    def test_the_host_mirrors_of_the_mask_kernel_agree_at_six_head_rows(self):
        """extent_attention_replay_tp.narrow_mask_host against this harness's own served_mask, for every R1 geometry at
        every word: two transliterations of attention_mask_replay_tp.cpp that must agree before either is an oracle."""
        report = dict(failures=[])
        with four():
            for name, (rows, batches, offset) in reader_b.R1_GEOMETRIES.items():
                self.assertTrue(reader_b.mirrors_agree(torch, extent_tp, rows, batches, offset, range(256), report, name,
                                                       QUAD), name)
            self.assertEqual(report['failures'], [])
            with mock.patch.object(extent_tp, 'narrow_mask_host', lambda word, *a: reader_b.served_mask(
                    torch, word + 1, 256, a[0], a[1], a[2], QUAD)):
                self.assertFalse(reader_b.mirrors_agree(torch, extent_tp, 8, 2, 0, [7], report, 'G8B2', QUAD))
            self.assertIn('disagree at word 7', report['failures'][0])
            # The wide mask of a ticket at `start`: the twin's replay_mask_host at capacity E, bit for bit.
            for start, extent in ((128, 256), (700, 1024), (2200, 2304)):
                wide = reader_b.wide_mask(torch, start, extent, QUAD)
                self.assertEqual(tuple(wide.shape), (2, 1, 8 * 6, extent))
                self.assertEqual(card.differing(torch, wide, extent_tp.replay_mask_host(start, 8, 2, 0, extent)), 0)
                # The semantics, not a transliteration: row h of entry b is token h // 6 of the group at position
                # start + 8 b + token; every cache position in the last chunk past it is -inf, every other +0.0.
                for batch in range(2):
                    for head in (0, 5, 6, 47):
                        position = start + batch * 8 + head // 6
                        for column in (extent - 256, extent - 1):
                            want = float('-inf') if column > position else 0.0
                            self.assertEqual(float(wide[batch, 0, head, column]), want, (start, batch, head, column))
                self.assertTrue(bool((wide[..., :extent - 256] == 0).all()))
        # The pair's is k64j_card_b's own.
        self.assertEqual(card.differing(torch, reader_b.served_mask(torch, 7, 256, 8, 2, 0),
                                        card_b.served_mask(torch, 7, 256, rows=8, batches=2, offset=0)), 0)
        self.assertEqual(tuple(reader_b.wide_mask(torch, 300, 512).shape), (2, 1, 96, 512))

    def test_fold_and_unfold_at_six_head_rows_are_the_kernels_index_maps(self):
        tokens = torch.randn(1, 16, 6, 256).to(torch.bfloat16)
        folded = reader_b.fold_entries(torch, tokens, (0, 8), 8, QUAD)
        self.assertEqual(tuple(folded.shape), (1, 2, 48, 256))
        with four():
            for entry, offset in enumerate((0, 8)):
                flat = tokens[:, offset:offset + 8].reshape(48, 256)
                for row in range(48):
                    self.assertTrue(torch.equal(folded[0, entry, row], flat[attention_fold_dma_tp.source_row(8, row)]))
        self.assertTrue(torch.equal(reader_b.unfold_entries(torch, folded, 8, QUAD), tokens))
        padded = torch.cat([folded, torch.zeros(1, 2, 16, 256, dtype=folded.dtype)], dim=2)      # the tile-padded output
        self.assertTrue(torch.equal(reader_b.unfold_entries(torch, padded, 8, QUAD), tokens))
        wide = torch.randn(1, 16, 12, 256).to(torch.bfloat16)                   # the pair's are card's own
        self.assertTrue(torch.equal(reader_b.fold_entries(torch, wide, (0, 8), 8),
                                    card.fold_entries(torch, wide, (0, 8), 8)))
        self.assertTrue(torch.equal(reader_b.unfold_entries(torch, reader_b.fold_entries(torch, wide, (0, 8), 8), 8), wide))

    def test_a_token_query_is_six_heads_on_the_one_kv_head(self):
        normal = reader_b.token_query(torch, 0, 'normal', 130, geometry=QUAD)
        self.assertEqual((tuple(normal.shape), normal.dtype), ((6, 256), torch.bfloat16))
        self.assertTrue(torch.equal(normal, reader_b.token_query(torch, 0, 'normal', 130, geometry=QUAD)))
        self.assertFalse(torch.equal(normal, reader_b.token_query(torch, 0, 'normal', 131, geometry=QUAD)))
        generator = torch.Generator().manual_seed(5)
        keys = torch.randn(10, 1, 64, 256, generator=generator).to(torch.bfloat16)          # ONE KV head
        table = torch.randperm(10, generator=generator).to(torch.int32)
        peaky = reader_b.token_query(torch, 0, 'peaky', 130, keys=keys, table=table, geometry=QUAD)
        self.assertEqual(tuple(peaky.shape), (6, 256))
        self.assertFalse(torch.equal(peaky, normal))
        with self.assertRaises(ValueError):
            reader_b.token_query(torch, 0, 'peaky', 130, geometry=QUAD)
        with self.assertRaises(ValueError):
            reader_b.token_query(torch, 0, 'zeroq', 130, geometry=QUAD)
        # A pair pool's second KV head is never read: the lookup stays on KV head 0.
        self.assertTrue(torch.equal(peaky, reader_b.token_query(torch, 0, 'peaky', 130, geometry=QUAD,
                                                                 keys=torch.cat([keys, keys * 7], dim=1), table=table)))
        # The pair's query is k64j_card_b's own, bit for bit.
        keys2 = torch.randn(10, 2, 64, 256, generator=generator).to(torch.bfloat16)
        for variant in ('normal', 'peaky'):
            self.assertTrue(torch.equal(reader_b.token_query(torch, 1, variant, 200, keys=keys2, table=table),
                                        card_b.token_query(torch, 1, variant, 200, keys=keys2, table=table)))

    def test_the_one_kv_head_pool(self):
        fake = FakeReaderTtnn(torch)
        report = dict(_requested=set(), failures=[])
        pool = reader_b.make_pool(fake, torch, fake, 512, 3, 4, report, QUAD)
        self.assertIsInstance(pool, reader_b.OneKvHeadPool)
        self.assertIsInstance(pool, card_b.ExtentPool)
        self.assertEqual((tuple(pool.keys.shape), pool.kv_heads, len(pool.tables), len(pool.poison)),
                         ((512 // 64 + probe.POISON_BLOCKS, 1, 64, 256), 1, 4, probe.POISON_BLOCKS))
        self.assertEqual((tuple(pool.k.shape), tuple(pool.v.shape)), ((8 + probe.POISON_BLOCKS, 1, 64, 256),) * 2)
        self.assertTrue(bool((pool.keys[8:] == probe.POISON_K).all()))
        self.assertEqual(sorted(pool.tables[0].tolist()), list(range(8)))
        pool.close()
        pair = reader_b.make_pool(fake, torch, fake, 512, 3, 4, report, PAIR)
        self.assertIs(type(pair), card_b.ExtentPool)
        self.assertEqual(tuple(pair.keys.shape), (8 + probe.POISON_BLOCKS, 2, 64, 256))
        pair.close()

    def test_the_pindiag_lines_at_width_four(self):
        engaged = '[PINDIAG] extent replay engaged segments=4 flags=0x23,0x23,0x23,0x23 mask=narrow capacity=4352'
        mode = ("[PINDIAG] sdpa qwen-modes modes=extent,share,tail rows=16 capacity=4352 bundles=[2] "
                "flags=['0x23'] mask=narrow")
        self.assertEqual(reader_b.pindiag_problems([mode] * 4 + [engaged], 4352, geometry=QUAD), [])
        # The four-card lines are not the pair's and the pair's are not four-card evidence.
        self.assertEqual(len(reader_b.pindiag_problems([mode] * 4 + [engaged], 4352)), 2)
        pair_lines = [line.replace('0x23', '0x27').replace('extent,share,tail', 'extent,share,slice,tail')
                      for line in [mode] * 4 + [engaged]]
        self.assertEqual(len(reader_b.pindiag_problems(pair_lines, 4352, geometry=QUAD)), 2)
        self.assertEqual(reader_b.pindiag_problems(pair_lines, 4352), [])
        self.assertEqual(len(reader_b.pindiag_problems([mode] * 4 + [engaged.replace('0x23,0x23', '0x3,0x23')], 4352,
                                                       geometry=QUAD)), 1)
        self.assertIn("flags=['0x23']", reader_b.pindiag_problems([engaged], 4352, geometry=QUAD)[0])
        # 0x23 reached with the pair image's modes (the slice dropped at one KV head) is not how four cards serve it.
        sliced = [line.replace('extent,share,tail', 'extent,share,slice,tail') for line in [mode] * 4]
        self.assertEqual(len(reader_b.pindiag_problems(sliced + [engaged], 4352, geometry=QUAD)), 1)

    def test_the_four_card_modes_are_the_four_card_profiles(self):
        """CB2b-TP4 runs the reader with the QWEN_FAST_SDPA_MODES every four-card S2 profile serves (tail,share), which
        gives the twin's 0x23; the pair keeps the image's tail,share,slice."""
        profiles = json.loads(read(CI / 'qwen_c2_profiles.json'))['profiles']
        served = {name: profile['env'].get(reader_b.SDPA_MODES_ENV) for name, profile in profiles.items()
                  if isinstance(profile, dict) and (profile.get('env') or {}).get(reader_b.TP_ENV) == '4'
                  and (profile.get('env') or {}).get('QWEN_FAST_EXTENT_REPLAY') == '1'}
        self.assertIn('c2-packed-tp4', served)
        self.assertEqual(set(served.values()), {QUAD.modes_env}, served)
        self.assertEqual((QUAD.modes_env, QUAD.modes_logged, QUAD.sdpa_modes), ('tail,share', 'extent,share,tail',
                                                                                 ('share', 'tail')))
        self.assertEqual((PAIR.modes_env, PAIR.modes_logged, PAIR.sdpa_modes),
                         ('tail,share,slice', 'extent,share,slice,tail', reader_b.SDPA_MODES))
        with four():
            self.assertEqual(pooled_attention_replay.mode_flags(set(QUAD.sdpa_modes) | {'extent'}, 2, 8),
                             QUAD.served_flags)

    def test_the_verdict_line_at_width_four(self):
        report = self.full_report()
        report.update(width=4, heads=6, kv_heads=1, chip_view=dict(chips_presented=4, phantom_programs=30),
                      modules=dict(sha256={'extent_attention_replay_tp.py': 'cd' * 32,
                                           'extent_attention_replay.py': 'ab' * 32}))
        report['decision'] = reader_b.decide(report)
        line = reader_b.verdict_line(report)
        self.assertTrue(line.startswith('K64J_READER verdict=PASS scope=full r1=2/2 r1_reader=1/1 staging=6/6 r2=2/2 '
                                        'r2_trace=1/1 r4=3/3 live=1/1 families=56 chips=1of4 phantom=30 '
                                        'extent_sha256=' + 'cd' * 32), line)
        words = recorder.line_words(line)
        self.assertEqual((words['chips'], words['phantom'], words['scope'], words['extent_sha256']),
                         ('1of4', '30', 'full', 'cd' * 32))
        # The same report without the width is the pair's line (its sha, its chips).
        report.pop('width')
        report.pop('chip_view')
        report['two_chip_view'] = dict(chips_presented=2, phantom_programs=30)
        line = reader_b.verdict_line(report)
        self.assertIn(' chips=1of2 phantom=30 extent_sha256=' + 'ab' * 32, line)


class QuadContractTests(unittest.TestCase):
    def test_the_four_chip_view_is_the_two_chip_views_contract_at_four_shards(self):
        fake = FakeReaderTtnn(torch)
        view = chip_view.ChipView(fake, chips=QUAD.chips)
        tensor = fake.from_torch(torch.zeros(8, dtype=torch.int32), dtype=fake.int32, layout=fake.ROW_MAJOR_LAYOUT,
                                 device=fake)
        self.assertEqual([shard.buffer_address() for shard in view.get_device_tensors(tensor)], [tensor.address] * 4)
        program = view.MeshProgramDescriptor()
        for chip in range(4):
            coordinate = view.MeshCoordinate(0, chip)
            program[view.MeshCoordinateRange(coordinate, coordinate)] = 'chip%d' % chip
        with self.assertRaises(ValueError):
            program[view.MeshCoordinateRange((0, 4), (0, 4))] = 'chip4'
        with self.assertRaises(ValueError):
            program[view.MeshCoordinateRange((0, 3), (0, 3))] = 'again'
        real = program.realise()
        self.assertEqual(dict(real), {ONE: 'chip0'})
        self.assertEqual((view.realised, view.phantom), (1, 3))
        half = view.MeshProgramDescriptor()
        for chip in (0, 1):
            half[view.MeshCoordinateRange((0, chip), (0, chip))] = 'chip%d' % chip
        with self.assertRaisesRegex(ValueError, 'every chip'):
            half.realise()
        two = mock.Mock(get_device_tensors=lambda value: [Shard(1), Shard(2)])
        with self.assertRaisesRegex(RuntimeError, 'ONE chip'):
            chip_view.ChipView(two, chips=4).get_device_tensors(tensor)
        saved = sys.modules.get('ttnn')
        with view.installed():
            self.assertIs(sys.modules['ttnn'], view)
        self.assertIs(sys.modules.get('ttnn'), saved)

    def test_without_the_view_the_real_twin_refuses_one_chip(self):
        fake = FakeReaderTtnn(torch)
        positions = extent_tp._upload(fake, fake, torch.zeros(8, dtype=torch.int32), fake.int32)
        mask = extent_tp._upload(fake, fake, torch.zeros(2, 1, 48, 256, dtype=torch.bfloat16), fake.bfloat16)
        with four(), mock.patch.dict(sys.modules, {'ttnn': fake}):
            with self.assertRaisesRegex(ValueError, 'Four chip-local metadata buffers required'):
                extent_tp.prepare_narrow(fake, positions, mask, rows=8, batches=2, offset=0)

    def test_the_fake_kernels_are_the_siblings_mirrors_at_six_head_rows(self):
        """The fake's emulations of attention_mask_replay_tp.cpp and attention_fold_dma_tp.cpp agree with the twin's host
        mirror and with this harness's fold, so the device flow below tests the reader, not the fake."""
        fake = FakeQuadTtnn(torch)
        with four():
            view = chip_view.ChipView(fake, chips=4)
            for rows, batches, offset in reader_b.R1_GEOMETRIES.values():
                positions = extent_tp._upload(view, fake, torch.zeros(8, dtype=torch.int32), view.int32)
                mask = extent_tp._upload(view, fake, torch.zeros(batches, 1, rows * 6, 256, dtype=torch.bfloat16),
                                         view.bfloat16)
                with view.installed():
                    program = extent_tp.prepare_narrow(fake, positions, mask, rows=rows, batches=batches, offset=offset)
                for word in (0, 7, 32, 200, 255):
                    fake.memory[positions.address] = torch.tensor([word] + [0] * 7, dtype=torch.int32)
                    with view.installed():
                        extent_tp.attention_mask_replay.execute(positions, mask, program)
                    self.assertEqual(card.differing(torch, fake.memory[mask.address],
                                                    extent_tp.narrow_mask_host(word, rows, batches, offset)), 0)
            tokens = torch.randn(1, 16, 6, 256).to(torch.bfloat16)
            source = fake.from_torch(tokens, dtype=fake.bfloat16, layout=fake.TILE_LAYOUT, device=fake)
            owned = []
            with view.installed():
                folded = attention_fold_dma_tp.device_layout_dma(fake, source, 8, owned, offset=8)
            self.assertTrue(torch.equal(fake.memory[folded.address],
                                        reader_b.fold_entries(torch, tokens, (8,), 8, QUAD)))
            with view.installed():
                back = attention_fold_dma_tp.device_layout_dma(fake, folded, 8, owned, inverse=True)
            self.assertTrue(torch.equal(fake.memory[back.address], tokens[:, 8:16]))
        names = {name for name, _tasks in fake.launched}
        self.assertEqual(names, {'attention_mask_replay_tp.cpp', 'attention_fold_dma_tp.cpp'})

    def test_a_sibling_built_without_its_define_is_refused_by_the_fake(self):
        """The fake checks what the device would: a _tp kernel without QWEN_FOLD_HEAD_ROWS does not compile."""
        fake = FakeQuadTtnn(torch)
        with four():
            view = chip_view.ChipView(fake, chips=4)
            positions = extent_tp._upload(view, fake, torch.zeros(8, dtype=torch.int32), view.int32)
            mask = extent_tp._upload(view, fake, torch.zeros(2, 1, 48, 256, dtype=torch.bfloat16), view.bfloat16)
            with view.installed(), mock.patch.object(extent_tp.tp_kernels, 'fold_defines', lambda environ=None: []):
                program = extent_tp.prepare_narrow(fake, positions, mask, rows=8, batches=2, offset=0)
            with view.installed(), self.assertRaisesRegex(RuntimeError, 'built with defines'):
                extent_tp.attention_mask_replay.execute(positions, mask, program)


class QuadFlowTests(FlowBase):
    """The harness at width 4 end to end on the fake one-chip ttnn: the REAL twin classes (extent_attention_replay_tp)
    through ChipView(chips=4), at 0x23 on one KV head, against the 0x3 compile-time call. The fake's K64j SDPA is
    FakeExtentTtnn's at one KV head (card.KV_HEADS, the cores per head)."""

    WIDTH_ARGV = ('--width', '4')
    WIDTH_ENV = {reader_b.TP_ENV: '4', 'QWEN_FAST_SDPA_MODES': 'tail,share'}      # what run_card_b.sh TP4_WIDTH=4 sets

    def width_patches(self):
        original = model.cores_per_head
        return [mock.patch.object(card, 'KV_HEADS', 1),
                mock.patch.object(model, 'cores_per_head',
                                  lambda batches, kv_heads=1, *args, **kwargs: original(batches, kv_heads, *args, **kwargs)),
                mock.patch.object(extent_tp, 'print', create=True)]

    def test_pass_end_to_end(self):
        fake = FakeQuadTtnn(torch)
        status, report = self.run_reader(fake)
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict']), (0, True, 'PASS'))
        self.assertEqual(report['sections_done'], ['R1/seed0', 'block/seed0'])
        kinds = self.kinds(report)
        # The same comparisons as the pair's run: the plan does not depend on the width.
        self.assertEqual(kinds['r1_eager'], (15, 15))
        self.assertEqual(kinds['r1_trace'], (15, 15))
        self.assertEqual(kinds['r1_reader'], (32, 32))
        self.assertEqual((kinds['staging_word'], kinds['staging_cur_pos'], kinds['staging_table']), ((4, 4),) * 3)
        self.assertEqual((kinds['restage_word'], kinds['restage_table']), ((40, 40), (24, 24)))
        self.assertEqual((kinds['r2_construction_vs_wide'], kinds['r2_trace_vs_wide'], kinds['r2_trace_vs_eager']),
                         ((4, 4), (32, 32), (8, 8)))
        self.assertEqual((kinds['r4_live_unchanged'], kinds['r4_idle_finite'], kinds['r4_idle_vs_wide']),
                         ((16, 16), (8, 8), (8, 8)))
        self.assertTrue(report['liveness'] and all(entry['live'] for entry in report['liveness']), report['liveness'])
        self.assertEqual(report['r2_families_replayed'], [256, 512, 768, 1280, 2048, 2304])
        # What the recorder reads of the report: the width, the flags, the reader's sha, the siblings' shas, chips=1of4.
        self.assertEqual((report['width'], report['heads'], report['kv_heads'], report['env'][reader_b.TP_ENV]),
                         (4, 6, 1, '4'))
        self.assertIs(type(report['kv_heads']), int)
        self.assertEqual({key: report['served'][key] for key in ('flags', 'reference_flags', 'rows', 'batch')},
                         dict(flags='0x23', reference_flags='0x3', rows=8, batch=2))
        self.assertEqual(len(report['segments']), 4)
        self.assertEqual(report['extent_reader'], dict(flags=['0x23'] * 4, segments=4, capacity=2304, borrowed=8))
        self.assertIn([0x23, 2, 2304 // 32, 8], report['requested_programs'])
        self.assertNotIn(0x27, {row[0] for row in report['requested_programs']})
        for family in (256, 512, 768, 1280, 2048, 2304):
            self.assertIn([0x3, 2, family // 32, family // 32], report['requested_programs'])
        self.assertEqual([(line['entries'], line['kv_share'], line['q_slice']) for line in report['extent_lines']],
                         [(2, 'true', 'false')])
        engaged = [line for line in report['pindiag'] if line.startswith(reader_b.ENGAGED_MARKER)]
        self.assertEqual(engaged, ['%s segments=4 flags=0x23,0x23,0x23,0x23 mask=narrow capacity=2304'
                                   % reader_b.ENGAGED_MARKER])
        # Served as the four-card profiles serve it: tail,share (no slice to drop), extent added by the reader.
        modes = [line for line in report['pindiag'] if line.startswith(reader_b.MODES_MARKER + ' modes=')]
        self.assertEqual(len(modes), 4)
        self.assertTrue(all(' modes=extent,share,tail ' in line and "flags=['0x23']" in line for line in modes), modes)
        self.assertEqual(report['env']['QWEN_FAST_SDPA_MODES'], 'tail,share')
        self.assertNotIn('two_chip_view', report)
        view = report['chip_view']
        self.assertEqual((view['chips_physical'], view['chips_presented']), (1, 4))
        self.assertEqual(view['phantom_programs'], 3 * view['programs_realised'])
        self.assertGreater(view['launches'], 100)
        shas = report['modules']['sha256']
        self.assertEqual(shas[recorder.READER_TP], sha((CI / recorder.READER_TP).read_bytes()))
        for name in recorder.PINNED_SIBLINGS:
            self.assertEqual(shas[name], sha((CI / name).read_bytes()), name)
        self.assertEqual(report['sibling_drift'], [])
        self.assertEqual(report['modules']['files']['extent_attention_replay_tp'], str((CI / recorder.READER_TP).resolve()))
        line = report['verdict_line']
        self.assertTrue(line.startswith('K64J_READER verdict=PASS scope=reduced r1=30/30 r1_reader=32/32 staging=116/116 '
                                        'r2=36/36 r2_trace=8/8 r4=32/32 live=4/4 families=6 chips=1of4 phantom='), line)
        words = recorder.line_words(line)
        self.assertEqual((words['chips'], words['extent_sha256'], int(words['phantom'])),
                         ('1of4', shas[recorder.READER_TP], view['phantom_programs']))
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))
        self.assertIn(('attention_mask_replay_tp.cpp', 32), fake.launched)      # 2 entries x 2 head tiles x 8 column tiles
        self.assertIn(('attention_fold_dma_tp.cpp', 64), fake.launched)
        self.assertFalse({name for name, _tasks in fake.launched} & {'attention_mask_replay.cpp', 'attention_fold_dma.cpp'},
                         'the pair\'s kernels never run at four cards')

    def test_a_stale_cur_pos_fails(self):
        def wrapper(original):
            def stage_values(self, start, table):
                values = original(self, start, table)
                if self.start is None:
                    return values
                return [value for value in values if not any(value[0] is positions for positions in self.cur_pos)]
            return stage_values

        original = extent_tp.ExtentSegmentReader.stage_values
        status, report = self.run_reader(FakeQuadTtnn(torch), base=self.SMALL, extra=['--sections', 'S,R2'],
                                         patches=[mock.patch.object(extent_tp.ExtentSegmentReader, 'stage_values',
                                                                    wrapper(original))])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.failing(report), {'restage_cur_pos', 'r2_trace_vs_wide'})

    def test_a_wide_mask_read_fails_r2_only(self):
        status, report = self.run_reader(FakeQuadTtnn(torch, broken={'tail_at_capacity'}), base=self.SMALL,
                                         extra=['--sections', 'R1,S,R2'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        failing = self.failing(report)
        self.assertLessEqual({'r2_construction_vs_wide', 'r2_trace_vs_wide'}, failing)
        self.assertFalse(failing & {'r1_eager', 'r1_trace', 'r1_reader', 'staging_word', 'staging_cur_pos'})

    def test_a_mask_kernel_that_skips_a_tile_fails_r1(self):
        status, report = self.run_reader(FakeQuadTtnn(torch, broken={'mask_stale_tile'}), base=self.SMALL,
                                         extra=['--sections', 'R1'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'FAIL'))
        self.assertLessEqual({'r1_eager', 'r1_trace'}, self.failing(report))

    def test_an_absolute_word_fails(self):
        def absolute(start):
            return start, extent_module.extent(start) - 1

        status, report = self.run_reader(FakeQuadTtnn(torch), base=self.SMALL, extra=['--sections', 'R1,S,R2'],
                                         patches=[mock.patch.object(extent_tp, 'extent_values', absolute)])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertLessEqual({'staging_word', 'restage_word', 'r1_reader', 'r2_trace_vs_wide'}, self.failing(report))

    def test_a_launch_that_did_not_carry_the_width_decides_nothing(self):
        """--width 4 with the process at the pair's width (QWEN_FAST_TP 2): tp_shapes would build twelve-row kernels.
        Refused before the device opens, with no comparison."""
        status, report = self.run_reader(FakeQuadTtnn(torch), base=self.SMALL, env={reader_b.TP_ENV: '2'})
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any('tp_shapes reads the width' in failure and 'read the launched argv' in failure
                            for failure in report['failures']), report['failures'])
        self.assertEqual(report['comparisons'], [])

    def test_the_sdpa_modes_and_the_scratch_are_checked_first_at_four_cards_too(self):
        # The pair image's modes are not the four-card profiles': refused, although they would give 0x23 at one KV head.
        for env, needle in (({'QWEN_FAST_SDPA_MODES': 'tail,share,slice'}, 'QWEN_FAST_SDPA_MODES must be tail,share,'),
                            ({'QWEN_FAST_SDPA_MODES': 'tail'}, 'QWEN_FAST_SDPA_MODES must be tail,share,'),
                            ({card.SCRATCH_ENV: '0'}, 'QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required')):
            with self.subTest(env=env):
                status, report = self.run_reader(FakeQuadTtnn(torch), base=self.SMALL, env=env)
                self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
                self.assertTrue(any(needle in failure for failure in report['failures']), report['failures'])
                self.assertEqual(report['comparisons'], [])

    def test_the_code_under_test_must_come_from_the_ci_root(self):
        status, report = self.run_reader(FakeQuadTtnn(torch), base=self.SMALL,
                                         extra=['--sections', 'S', '--ci-root', str(self.dir)])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any('was loaded from' in failure for failure in report['failures']), report['failures'])
        self.assertEqual(report['comparisons'], [])


@unittest.skipUnless(BASH, 'bash not found')
class QuadRunnerTests(unittest.TestCase):
    """run_card_b.sh K64J_HARNESS=extent_reader TP4_WIDTH=4 (dry run): the launch carries the width both ways, the
    four-card sources are required, the pair's run is what it was, and the evidence job templates call the harness
    with flags it implements and with the scopes the admission needs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.graft = card_tests.make_graft(self.dir)
        self.helper = RunnerTests('test_the_extent_readers_dry_run')
        self.helper.dir = self.dir
        self.helper.graft = self.graft

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, runner=RUNNER, **env):
        return self.helper.run_runner(runner, **env)

    def env_of(self, argv):
        return [argv[i + 1] for i, word in enumerate(argv) if word == '-e']

    def test_the_four_card_dry_run(self):
        result = self.run_runner(TP4_WIDTH='4')
        argv = self.helper.argv(result)
        self.assertEqual(result.stderr, '')
        env = self.env_of(argv)
        for value in ('QWEN_FAST_TP=4', 'QWEN_FAST_SDPA_MODES=tail,share', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS=1'):
            self.assertIn(value, env)
        self.assertEqual([value for value in env if value.startswith('QWEN_FAST_SDPA_MODES=')],
                         ['QWEN_FAST_SDPA_MODES=' + QUAD.modes_env], 'the four-card profiles\' modes, set once')
        args = self.helper.harness_args(argv)
        self.assertEqual((args.width, args.geometry, args.ci_root, args.served_root, args.sections, len(args.families)),
                         (4, QUAD, '/bench/ci', '/experiment-scripts/ci', ['R1', 'S', 'R2', 'R4'], 56))
        inner = argv[argv.index('--entrypoint') + 4]
        self.assertTrue(inner.endswith('exec python3 -B /bench/extent_reader_card_b.py "$@"'), inner)
        logged = inner.split('2>&1; ')[0]
        for name in tuple(reader_b.PINNED) + reader_b.QUAD_SIBLINGS:          # the image's copies, logged before the run
            self.assertIn('/experiment-scripts/ci/%s ' % name, logged)
        # Nothing of the image's is mounted; the code under test is this checkout's scripts/ci alone.
        self.assertNotIn('/experiment-scripts/ci', ' '.join(m.get('src', '') for m in self.helper.mounts(argv)))
        bench = {m['dst'] for m in self.helper.mounts(argv) if m['dst'].startswith('/bench/')}
        self.assertEqual(bench, {'/bench/ci', '/bench/extent_reader_card_b.py', '/bench/k64j_card_b.py',
                                 '/bench/probe_k1_card_b.py', '/bench/probe_k64j_card_b.py', '/bench/split_model.py',
                                 '/bench/test_sdpa_decode_qwen_card_m.py'})
        # The harness's own flags are the ones after the image: --width 4 comes from the width, never from CARD_B_ARGS.
        self.assertIn('--width', argv[argv.index('card') + 1:])
        watcher = self.helper.harness_args(self.helper.argv(self.run_runner(TP4_WIDTH='4', WATCHER='1')))
        self.assertEqual((watcher.width, watcher.seeds, watcher.variants, len(watcher.families)),
                         (4, [0], ['peaky'], 5))
        self.assertIn('TT_METAL_WATCHER=5', self.env_of(self.helper.argv(self.run_runner(TP4_WIDTH='4', WATCHER='1'))))

    def test_the_pairs_run_is_what_it_was(self):
        for width in ({}, {'TP4_WIDTH': '2'}):
            with self.subTest(width=width):
                argv = self.helper.argv(self.run_runner(**width))
                self.assertNotIn('--width', argv[argv.index('card') + 1:])
                self.assertFalse([value for value in self.env_of(argv) if value.startswith('QWEN_FAST_TP')])
                self.assertEqual([value for value in self.env_of(argv) if value.startswith('QWEN_FAST_SDPA_MODES=')],
                                 ['QWEN_FAST_SDPA_MODES=tail,share,slice'])
                self.assertEqual(self.helper.harness_args(argv).width, 2)
                inner = argv[argv.index('--entrypoint') + 4]
                self.assertFalse([name for name in reader_b.QUAD_SIBLINGS if name in inner])
        # The other harnesses never see the width argument: TP4_WIDTH is gdn_tp4's and extent_reader's alone.
        argv = self.helper.argv(self.run_runner(K64J_HARNESS='card', TP4_WIDTH='4'))
        self.assertNotIn('--width', argv)
        self.assertFalse([value for value in self.env_of(argv) if value.startswith('QWEN_FAST_TP')])

    def test_a_width_that_is_neither_is_refused_before_anything_is_launched(self):
        for value in ('3', '1', 'four'):
            result = self.run_runner(TP4_WIDTH=value)
            self.assertEqual(result.returncode, 1, value)
            self.assertIn('refusing: TP4_WIDTH=%s is neither 4 nor 2' % value, result.stderr)
            self.assertNotIn('### argv: ', result.stdout)

    def test_the_four_card_sources_are_required_and_the_pairs_run_needs_none_of_them(self):
        runner, ci = make_tree(self.dir)
        for name in reader_b.QUAD_RECORDED_SOURCES:
            self.assertTrue((ci / name).is_file(), name)
        self.helper.argv(self.run_runner(runner, TP4_WIDTH='4'))
        for name in ('extent_attention_replay_tp.py', 'chip_view.py', 'tp_shapes.py', 'tp_kernels.py', 'tp_addresses.py'):
            (ci / name).rename(ci / (name + '.moved'))
            result = self.run_runner(runner, TP4_WIDTH='4')
            self.assertEqual(result.returncode, 1, name)
            self.assertIn('%s missing (the extent reader runs this checkout' % name, result.stderr, name)
            self.helper.argv(self.run_runner(runner))                      # the pair's run does not need it
            (ci / (name + '.moved')).rename(ci / name)
        self.helper.argv(self.run_runner(runner, TP4_WIDTH='4'))

    def parse_template(self, name, **fill):
        values = template(name)
        self.assertEqual(values['C2_ACTIONS'], 'cardm')
        self.assertEqual(values['C2_CARDM_HARNESS'], 'optimisation/ttnn-op/k64j/run_card_b.sh')
        text = values['C2_CARDM_ENV']
        for placeholder, value in (('@K64J_GRAFT_DIR@', self.graft.as_posix()),
                                   ('@K64J_TTNNCPP_SHA256@', sha(card_tests.BINARY)),
                                   ('@SERVED_IMAGE@', 'tt-vllm:four-card-test')):
            text = text.replace(placeholder, value)
        self.assertNotIn('@', text)
        env = dict(pair.split('=', 1) for pair in text.split())
        env['CARD_B_ARGS'] = values['C2_CARDM_ARGS']
        return values, env

    def test_the_evidence_job_templates_call_the_harness_with_the_flags_it_implements(self):
        import c2_serving_job as job
        for name, watcher in (('EV-W2-cb2b-watcher.env', True), ('EV-F3-cb2b.env', False)):
            with self.subTest(template=name):
                values, env = self.parse_template(name)
                # The cardm step accepts the values (the workflow's own check).
                plain = values['C2_CARDM_ENV'].replace('@K64J_GRAFT_DIR@', '/home/thatch/opgraft-K64j').replace(
                    '@K64J_TTNNCPP_SHA256@', 'ab' * 32).replace('@SERVED_IMAGE@', 'tt-vllm:four-card-test')
                harness, words, pairs = job.read_cardm(dict(values, C2_CARDM_ENV=plain), True, root=str(ROOT))
                self.assertEqual((harness, words), (values['C2_CARDM_HARNESS'], values['C2_CARDM_ARGS']))
                self.assertEqual((env['K64J_HARNESS'], env['TP4_WIDTH'], env.get('WATCHER') == '1'),
                                 ('extent_reader', '4', watcher))
                result = self.run_runner(**env)
                argv = self.helper.argv(result)
                self.assertEqual(result.stderr, '')
                self.assertEqual(argv[argv.index('--entrypoint') + 2], 'tt-vllm:four-card-test',
                                 'the pinned siblings are served from the four-card image, not the P8 default')
                environment = self.env_of(argv)
                self.assertIn('QWEN_FAST_TP=4', environment)
                self.assertIn('QWEN_FAST_SDPA_MODES=tail,share', environment)
                self.assertEqual('TT_METAL_WATCHER=5' in environment, watcher)
                args = self.helper.harness_args(argv)       # every flag the template passes is one the harness parses
                self.assertEqual((args.width, args.sections, args.capacity), (4, ['R1', 'S', 'R2', 'R4'], 131328))
                if watcher:
                    self.assertEqual((args.seeds, args.variants), ([0], ['normal']))
                    self.assertLess(len(args.families), admission.CB2B_R2_MIN_FAMILIES,
                                    'a watcher pass is the reduced scope: never CB2b\'s evidence')
                    continue
                # The full job: what the admission needs, from the harness's defaults and the template's own flags.
                self.assertEqual((args.seeds, args.variants), (list(admission.CB2B_SEEDS), ['normal', 'peaky']))
                self.assertGreaterEqual(len(args.families), admission.CB2B_R2_MIN_FAMILIES)
                self.assertLessEqual(set(admission.CB2B_R2_NAMED), set(args.families))
                self.assertEqual(set(args.r1_geometries), set(admission.CB2B_R1_GEOMETRIES))
                self.assertLessEqual(set(admission.CB2B_RESIDUES), set(args.r1_words))
                self.assertLessEqual(set(admission.CB2B_RESIDUES), set(args.r2_residues))
                idle = {start for pattern in args.idle_patterns for start in reader_b.idle_assignment(pattern).values()}
                self.assertLessEqual(set(admission.CB2B_IDLE_STARTS), idle)
                self.assertNotIn('--capacity', values['C2_CARDM_ARGS'])
                # The scope the harness itself judges a run of these arguments to have covered is full.
                report = dict(capacity=args.capacity, sections=args.sections, seeds_run=args.seeds,
                              variants_run=args.variants, idle_starts_run=sorted(idle),
                              r1_run={name: list(args.r1_words) for name in args.r1_geometries},
                              r2_families_replayed=args.families)
                self.assertEqual(reader_b.scope(report), ('full', []))

    def test_the_template_header_comments_do_not_promise_a_flag_the_harness_lacks(self):
        for name in ('EV-W2-cb2b-watcher.env', 'EV-F3-cb2b.env'):
            text = read(TEMPLATES / name)
            self.assertIn('K64J_HARNESS=extent_reader TP4_WIDTH=4', text)
            self.assertNotIn('extent_reader_tp', text)


class QuadServedLoaderTests(unittest.TestCase):
    """load_modules at width 4 in a fresh interpreter, as the container runs it: the frozen pair modules and the twin's
    siblings from a served tree, the twin and its helpers from the checkout's scripts/ci, the width from the launch. The
    served tree holds DECOYS of tp_shapes and tp_kernels: the checkout's must be the ones the twin runs."""

    def setUp(self):
        import test_extent_attention_replay as extent_tests
        self.tmp = tempfile.TemporaryDirectory()
        self.served = Path(self.tmp.name) / 'experiment-scripts' / 'ci'
        self.served.mkdir(parents=True)
        with open(self.served / 'attention_mask_replay.py', 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(extent_tests.served_mask_source())
        for name in ('attention_mask_replay.cpp', 'attention_fold_dma.py', 'attention_fold_dma.cpp',
                     'frozen_context_geometry.py') + reader_b.QUAD_SIBLINGS:
            shutil.copyfile(CI / name, self.served / name)
        for name in ('tp_shapes.py', 'tp_kernels.py'):
            (self.served / name).write_text('DECOY = True' + NL, encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, width='4', prelude=''):
        script = NL.join((
            'import json, sys',
            'sys.path[:0] = %r' % [str(HERE), str(PROBE_DIR), str(OPS / 'sdpa_decode_qwen')],
            prelude,
            'import extent_reader_card_b as reader_b',
            'report = dict(failures=[], warnings=[])',
            'mods = reader_b.load_modules(%r, report, %r, reader_b.QUAD)' % (str(CI), str(self.served)),
            'out = dict(failures=report["failures"], warnings=report["warnings"], modules=report.get("modules"),',
            '           drift=report.get("sibling_drift"), loaded=mods is not None)',
            'if mods is not None:',
            '    out.update(bound=[mods.extent.attention_mask_replay is mods.mask,',
            '                      mods.extent.device_layout_dma is mods.fold.device_layout_dma,',
            '                      mods.extent_pair.attention_mask_replay is mods.mask,',
            '                      mods.extent_pair.device_layout_dma is mods.fold_pair.device_layout_dma,',
            '                      mods.pooled.validate_ticket is mods.mask.validate_ticket,',
            '                      sys.modules["attention_replay"].prepare is mods.mask.prepare],',
            '               tp_shapes=mods.tp_shapes.__file__, decoy=hasattr(mods.tp_shapes, "DECOY"),',
            '               kernels=mods.tp_kernels.__file__, view=mods.chip_view.__file__,',
            '               chips=mods.tp_shapes.chip_count(), cpp=mods.tp_kernels.source(%r))'
            % str(self.served / 'attention_mask_replay.cpp'),
            'print("LOADED " + json.dumps(out))'))
        environ = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        environ.pop(reader_b.TP_ENV, None)
        if width is not None:
            environ[reader_b.TP_ENV] = width
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=300, env=environ)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith('LOADED ')]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return json.loads(lines[0][len('LOADED '):])

    def test_the_twin_and_its_helpers_are_the_checkouts_and_the_pair_modules_and_siblings_the_served_trees(self):
        out = self.load()
        self.assertEqual((out['failures'], out['warnings'], out['loaded']), ([], [], True))
        self.assertEqual(out['bound'], [True] * 6)
        files = {name: Path(path) for name, path in out['modules']['files'].items()}
        for name in reader_b.QUAD_SERVED_MODULES:
            self.assertEqual(files[name], (self.served / (name + '.py')).resolve(), name)
        for _key, name in reader_b.QUAD_MODULES:
            if name not in reader_b.QUAD_SERVED_MODULES:
                self.assertEqual(files[name].parent, CI.resolve(), name)
        # The decoys in the served tree were never run: the twin's tp_shapes / tp_kernels are the checkout's.
        self.assertEqual([Path(out['tp_shapes']).parent, Path(out['kernels']).parent, Path(out['view']).parent],
                         [CI.resolve()] * 3)
        self.assertFalse(out['decoy'])
        self.assertEqual(out['chips'], 4)
        shas = out['modules']['sha256']
        self.assertEqual({name: shas[name] for name in reader_b.PINNED}, reader_b.PINNED)
        for name in reader_b.QUAD_SIBLINGS:                     # the served bytes (here the checkout's own)
            self.assertEqual(shas[name], sha((self.served / name).read_bytes()), name)
        for name in reader_b.QUAD_RECORDED_SOURCES:
            self.assertEqual(shas[name], sha((CI / name).read_bytes()), name)
        self.assertEqual(out['drift'], [])
        # The sibling kernel the launch builders pick: the _tp sibling beside the served pinned .cpp.
        self.assertEqual(Path(out['cpp']), (self.served / 'attention_mask_replay_tp.cpp').resolve())

    def test_a_served_sibling_that_is_not_the_checkouts_is_recorded_and_warned_about_not_refused(self):
        with open(self.served / 'attention_mask_replay_tp.cpp', 'ab') as handle:
            handle.write(b'// drift\n')
        out = self.load()
        self.assertEqual((out['failures'], out['loaded']), ([], True))
        self.assertEqual(out['drift'], ['attention_mask_replay_tp.cpp'])
        self.assertEqual(len(out['warnings']), 1)
        self.assertIn('attention_mask_replay_tp.cpp', out['warnings'][0])
        self.assertEqual(out['modules']['sha256']['attention_mask_replay_tp.cpp'],
                         sha((self.served / 'attention_mask_replay_tp.cpp').read_bytes()))
        self.assertNotEqual(out['modules']['sha256']['attention_mask_replay_tp.cpp'],
                            sha((CI / 'attention_mask_replay_tp.cpp').read_bytes()))

    def test_the_frozen_pair_bytes_are_refused_at_four_cards_too(self):
        with open(self.served / 'attention_fold_dma.cpp', 'ab') as handle:
            handle.write(b'// drift\n')
        out = self.load()
        self.assertFalse(out['loaded'])
        self.assertTrue(any(failure.startswith('pinned source attention_fold_dma.cpp is ')
                            for failure in out['failures']), out['failures'])

    def test_a_launch_without_the_width_decides_nothing(self):
        for width in (None, '2'):
            out = self.load(width=width)
            self.assertFalse(out['loaded'], width)
            self.assertTrue(any('tp_shapes reads the width 2' in failure and 'read the launched argv' in failure
                                for failure in out['failures']), out['failures'])

    def test_a_missing_served_sibling_module_or_one_imported_first_decides_nothing(self):
        (self.served / 'attention_fold_dma_tp.py').unlink()
        out = self.load()
        self.assertFalse(out['loaded'])
        self.assertTrue(any(failure.startswith('attention_fold_dma_tp.py is not in --served-root ')
                            for failure in out['failures']), out['failures'])
        shutil.copyfile(CI / 'attention_fold_dma_tp.py', self.served / 'attention_fold_dma_tp.py')
        out = self.load(prelude='sys.path.insert(0, %r); import attention_fold_dma_tp' % str(CI))
        self.assertFalse(out['loaded'])
        self.assertTrue(any(failure.startswith('attention_fold_dma_tp was imported from ')
                            and '--served-root' in failure for failure in out['failures']), out['failures'])


if __name__ == '__main__':
    unittest.main()
