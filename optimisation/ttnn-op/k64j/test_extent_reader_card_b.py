"""CPU checks for W10b's CB2b harness (extent_reader_card_b.py; run_card_b.sh K64J_HARNESS=extent_reader); no device.

  - the helpers: the live starts (the floor, the table's end), the family plan (the named first, more than 50 by
    default), the replay plan (every family, the restages), the idle patterns, the decision, the scope and the
    verdict line, the PINDIAG check, the arguments;
  - the contract: the pinned shas are the frozen ones (test_extent_attention_replay's) and the runner's; the lent
    storage is what serving_buffer_pool's extent storage is, tensor for tensor; the two host mirrors of the mask kernel
    agree; the two-chip view (one shard twice, chip 0's program launched, chip 1's counted, sys.modules restored);
    without the view the real reader refuses a one-chip device;
  - the runner (needs bash): K64J_HARNESS=extent_reader's dry run (the harness and this checkout's scripts/ci mounted
    read-only, QWEN_FAST_SDPA_MODES, the report and container names), its watcher pass, an unknown harness and a
    changed pinned source refused before anything is launched, card M through the override (the dry run, and the real
    launch path on test_qual_card's fake rig), and the cardm job file's check of the documented values;
  - the whole flow on a fake ONE-chip ttnn (test_k64j_card_b.FakeExtentTtnn's K64j SDPA, read at replay time, plus
    the pinned mask and fold kernels emulated from their .cpp, slices, concats and traces): the REAL reader classes
    through the two-chip view, PASS end to end, and each broken variant on the section that must catch it - a stale
    cur_pos, an absolute word, a wide mask read, a missing slot copy (alone, and with a device that lacks R10's slot
    copy), an unstaged construction, a mask kernel that skips a tile, a trace that keeps its captured cur_pos, a
    non-finite idle row, a missing PINDIAG or F22 line, a module from elsewhere, the wrong QWEN_FAST_SDPA_MODES and
    the deadline.

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

import extent_attention_replay as extent_module  # noqa: E402
import pooled_attention_replay  # noqa: E402
import extent_reader_card_b as reader_b  # noqa: E402
import k64j_card_b as card_b  # noqa: E402
import split_model as model  # noqa: E402
import test_k64j_card_b as card_tests  # noqa: E402 - FakeExtentTtnn, make_graft, the runner's fake rig
import test_k64j_probe as probe_tests  # noqa: E402 - FakeTensor, HostTensor, ELEMENT, bash

probe = card_b.probe
card = probe.card
RUNNER = HERE / 'run_card_b.sh'
CARD_B, CARD_M = probe_tests.CARD_B, probe_tests.CARD_M
BASH = probe_tests.BASH
SCRUB = probe_tests.SCRUB + ('K64J_CARD_DRY_RUN', 'K64J_HARNESS', 'QWEN_FAST_SDPA_MODES')
NL = chr(10)
ONE = ('range', ('coord', 0, 0), ('coord', 0, 0))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


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
    NaN)."""

    class DataMovementProcessor:
        RISCV_0 = 'riscv0'

    class NOC:
        RISCV_0_default = 'noc0'

    def __init__(self, torch, *args, **kwargs):
        super().__init__(torch, *args, **kwargs)
        self.launched = []

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

        def write(replay=False):
            self.memory[output.address] = self.torch.cat([self.memory[address] for address in sources], dim=dim)

        self.run_or_record(write)
        return output

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
            run = {'attention_mask_replay.cpp': self.mask_launch,
                   'attention_fold_dma.cpp': self.fold_launch}.get(path.name)
            if run is None:
                raise RuntimeError('TT_FATAL: no emulation of %s' % path.name)
            launches.append((run, tasks))
            self.launched.append((path.name, len(tasks)))

        def write(replay=False):
            for run, tasks in launches:
                run(tasks, shapes)

        self.run_or_record(write)

    def mask_launch(self, tasks, shapes):
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
            word = int(self.memory[positions].reshape(-1)[0]) & 0xffffffff
            head_tiles = (rows * 12 + 31) // 32
            batch, head_tile, column_tile = task // (head_tiles * 8), (task // 8) % head_tiles, task % 8
            if 'mask_stale_tile' in self.broken and column_tile == 7:
                continue
            head = head_tile * 32 + torch.arange(32)
            position = word + offset + batch * rows + (head % (rows * 6)) // 6
            cache = capacity - 256 + column_tile * 32 + torch.arange(32)
            masked = (head[:, None] >= rows * 12) | (cache[None, :] > position[:, None])
            tile = torch.where(masked, float('-inf'), 0.0).to(torch.bfloat16)
            page = (batch * head_tiles + head_tile) * (capacity // 32) + capacity // 32 - 8 + column_tile
            block, column = divmod(page, width // 32)
            entry, row_tile = divmod(block, mask_row_tiles)
            if entry >= batches:
                raise RuntimeError('TT_FATAL: mask kernel page %d is past the %r mask' % (page, tuple(mask.shape)))
            first, last = row_tile * 32, min(row_tile * 32 + 32, heads)
            mask[entry, 0, first:last, column * 32:(column + 1) * 32] = tile[:last - first]
        self.memory[target] = mask

    def fold_launch(self, tasks, shapes):
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
            heads = torch.arange(12)
            folded = (heads // 6)[None, :] * rows * 6 + torch.arange(rows)[:, None] * 6 + (heads % 6)[None, :]
            full = data[0, 0][folded]                                            # (rows, 12, 256)
            cover = covered.repeat_interleave(32, dim=1)[:, None, :].expand(rows, 12, 256)
            output[0] = torch.where(cover, full, output[0])
        else:
            count = rows * 12
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
            data = torch.zeros(output.shape, dtype=torch.bfloat16)
            for entry in range(batches):
                word = words[entry if (not share or 'own_slot' in self.broken) else 0]
                row = table[0 if share else entry]
                if 'idle_nonfinite' in self.broken and word == 255 and not bool(row.any()):
                    data[0, entry] = float('nan')
                    continue
                result = self.attend(q[0, entry], keys, values, row, word, k_chunk // 32, scale, cores, False,
                                     mask[entry, 0], True, capacity)
                data[0, entry] = result.to(torch.bfloat16)
            self.memory[output.address] = data

        self.run_or_record(write)
        self.now += 20e-6 + 1e-9 * capacity if self.seconds_per_call is None else self.seconds_per_call
        return output


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
                    variants_run=['normal', 'peaky'], idle_starts_run=[0, 32], failures=[])

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

    def test_the_verdict_line(self):
        report = self.full_report()
        report['two_chip_view'] = dict(phantom_programs=12)
        report['decision'] = reader_b.decide(report)
        line = reader_b.verdict_line(report)
        self.assertTrue(line.startswith('K64J_READER verdict=PASS scope=full r1=2/2 r1_reader=1/1 staging=6/6 r2=2/2 '
                                        'r2_trace=1/1 r4=3/3 live=1/1 families=56 chips=1of2 phantom=12'), line)

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
    def test_the_pinned_shas_are_the_frozen_ones_and_the_runners(self):
        import test_extent_attention_replay as extent_tests
        frozen = extent_tests.StructureTests.SOURCES
        self.assertEqual(reader_b.PINNED, {name: frozen[name] for name in reader_b.PINNED})
        for name, digest in reader_b.PINNED.items():
            self.assertEqual(sha((CI / name).read_bytes()), digest, name)
        runner = read(RUNNER)
        for variable, name in (('MASK_PY', 'attention_mask_replay.py'), ('MASK_CPP', 'attention_mask_replay.cpp'),
                               ('FOLD_PY', 'attention_fold_dma.py'), ('FOLD_CPP', 'attention_fold_dma.cpp')):
            self.assertEqual(re.findall(r'^%s=([0-9a-f]{64})$' % variable, runner, flags=re.M), [reader_b.PINNED[name]])
        sources = re.search(r"^CI_SOURCES='([^']*)'$", runner, flags=re.M | re.S).group(1).split()
        self.assertEqual(sorted(sources), sorted(set(reader_b.PINNED) | set(reader_b.RECORDED_SOURCES)))
        self.assertEqual({name for _key, name in reader_b.MODULES}, {name[:-3] for name in sources
                                                                     if name.endswith('.py')})
        self.assertIn("exec python3 -B /bench/extent_reader_card_b.py \"$@\"", runner)
        self.assertEqual(reader_b.CI_ROOT, '/bench/ci')

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
    for name in set(reader_b.PINNED) | set(reader_b.RECORDED_SOURCES):
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
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-reader-card-b')
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
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
        args = self.harness_args(argv)
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.deadline_s, args.ci_root, args.sections,
                          len(args.families)), (sha(card_tests.BINARY), 300.0, 4800.0, '/bench/ci',
                                                ['R1', 'S', 'R2', 'R4'], 56))
        self.assertRegex(args.out.as_posix(), r'^/results/reader-[0-9]{8}T[0-9]{6}\.json$')
        # The default harness is untouched by the selection: no /bench/ci, no QWEN_FAST_SDPA_MODES.
        argv = self.argv(self.run_runner(K64J_HARNESS='card'))
        self.assertNotIn('/bench/ci', [m['dst'] for m in self.mounts(argv)])
        self.assertNotIn('QWEN_FAST_SDPA_MODES=tail,share,slice', argv)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-card-card-b')

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
        self.assertIn('refusing: K64J_HARNESS=k2 is neither card', result.stderr)
        self.assertNotIn('### argv: ', result.stdout)

    def tree(self):
        return make_tree(self.dir)

    def test_a_changed_pinned_source_or_a_missing_one_is_refused_before_launch(self):
        runner, ci = self.tree()
        self.argv(self.run_runner(runner))
        with open(ci / 'attention_fold_dma.cpp', 'ab') as handle:
            handle.write(b'// drift\n')
        result = self.run_runner(runner)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        drifted = sha((ci / 'attention_fold_dma.cpp').read_bytes())
        self.assertIn('attention_fold_dma.cpp is %s, not its frozen %s (a pinned source changed)'
                      % (drifted, reader_b.PINNED['attention_fold_dma.cpp']), result.stderr)
        self.assertNotIn('### argv: ', result.stdout)
        shutil.copyfile(CI / 'attention_fold_dma.cpp', ci / 'attention_fold_dma.cpp')
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
            'fuser() { local n=${@: -1}; case " ${FAKE_HELD:-} " in *" $n "*) '
            'echo "$n: thatch 4242 F.... python3" >&2; return 0 ;; esac; return 1; }',
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
        self.assertIn('refusing: host processes hold %s' % node, result.stderr)
        self.assertFalse(self.launched.exists())


# ---------------------------------------------------------------------------------------------
# The device flow on the fake one-chip ttnn, through the real reader classes.
# ---------------------------------------------------------------------------------------------

class DryRunTests(unittest.TestCase):
    """The harness end to end on the fake: a 2,304-key table (9 chunks), the families 256 / 512 / 2,304 and three
    spread ones. One FULL run covers every section; each broken variant runs only what must catch it, on a 1,280-key
    table whose five families make two assignments (SMALL)."""

    FULL = ['--capacity', '2304', '--r2-named', '256,512,2304', '--r2-families', '6', '--r2-residues',
            '0,7,240,255', '--seeds', '0', '--variants', 'normal,peaky', '--r1-words', '0,7,32,240,255',
            '--idle-patterns', '3,2+3,0']
    SMALL = ['--capacity', '1280', '--r2-named', '256,512,1280', '--r2-families', '5', '--r2-residues', '7,250',
             '--seeds', '0', '--variants', 'normal', '--r1-words', '7,255', '--r1-geometries', 'G8B2',
             '--r2-restages', '1', '--idle-patterns', '2+3']

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
        argv = ['--out', str(out), '--kernel-root', str(self.kernels), '--ci-root', '',
                '--expect-binary-sha256', sha(self.binary.read_bytes())]
        environ = {card.SCRATCH_ENV: '1', 'QWEN_FAST_SDPA_MODES': 'tail,share,slice'}
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
            stack.enter_context(mock.patch('sys.stdout'))
            stack.enter_context(mock.patch('pooled_attention_replay._binary_checked', []))
            stack.enter_context(mock.patch('pooled_attention_replay.loaded_binary_has_modes',
                                           return_value=('/k64j/_ttnncpp.so', True)))
            for patch in patches:
                stack.enter_context(patch)
            status = reader_b.main(argv + list(self.FULL if base is None else base) + list(extra))
        return status, json.loads(out.read_text())

    def kinds(self, report):
        return {kind: (row['equal'], row['runs']) for kind, row in report['tally'].items()}

    def failing(self, report):
        return {kind for kind, (equal, runs) in self.kinds(report).items() if equal != runs}

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
        self.assertEqual((report['variants_run'], report['idle_starts_run'], report['captures']),
                         (['normal', 'peaky'], [0, 32], 1))
        self.assertEqual(report['extent_reader'], dict(flags=['0x27'] * 4, segments=4, capacity=2304, borrowed=8))
        self.assertIn([0x27, 2, 2304 // 32, 8], report['requested_programs'])
        for family in (256, 512, 768, 1280, 2048, 2304):
            self.assertIn([0x7, 2, family // 32, family // 32], report['requested_programs'])
        self.assertEqual([(line['entries'], line['kv_share'], line['q_slice']) for line in report['extent_lines']],
                         [(2, 'true', 'true')])
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
        pinned = dict(reader_b.PINNED, **{'attention_mask_replay.cpp': '0' * 64})
        status, report = self.run_reader(FakeReaderTtnn(torch), base=self.SMALL, extra=['--sections', 'S'],
                                         patches=[mock.patch.object(reader_b, 'PINNED', pinned)])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any(failure.startswith('pinned source attention_mask_replay.cpp') for failure in
                            report['failures']), report['failures'])

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
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))


if __name__ == '__main__':
    unittest.main()
