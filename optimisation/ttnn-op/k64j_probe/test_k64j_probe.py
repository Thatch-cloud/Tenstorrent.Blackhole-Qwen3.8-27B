"""CPU checks for the K64j P0 probe (split_model.py, probe_k64j_card_b.py, run_card_b.sh); no device, no ttnn.

  - the split (R1 on paper): split_model mirrors rt_args_common.hpp and get_tree_reduction_params (the vendored
    fixtures, sha-pinned; with a host g++ on PATH or QWEN_PF_GXX the fixtures are COMPILED and every workload and
    tree answer compared), and over every 256-key family of the served 131,328-key geometry, every core count the
    probe or the replay uses and every cur_pos in [E - 256, E - 1] the runtime split IS the compile-time call's at
    capacity E; the fixed chunk ignores max_dynamic_chunk_size (served value 8); the stale-writer hazard is exactly
    E / 256 < cores per head, and every skip;
  - the flag bit: 0x20 is none of the flags any factory generator, the served reader, the gate or a card test uses
    (0x10 is the card tests' unknown-flag control);
  - the probe's helpers: the position plans, the poisoned table, the masks against the causal writer's
    generate_mask, the fold, the served modes per binary stage, the decision and the verdict line;
  - the contract: the runner's image, graft and binary are the C2 gate arm's (v235), and the geometry the profile's;
  - the runner (needs bash): the dry run on card B with the arm's graft mounts and the harness files, the watcher
    pass, KOPGRAFT64=none, the graft refusals, the verdict echo;
  - the whole flow on a fake ttnn with the op's semantics (the split and the tree of split_model, the causal and
    non-causal paths, the served flags, trace capture that reads the cur_pos tensor at replay, a DRAM allocator
    that reuses freed addresses): GO end to end, and each broken variant is caught - an extent read one chunk too
    far (NO-GO), a trace that ignores the rewritten cur_pos (NO-GO), a skip that disturbs a live entry (NO-GO), a
    split taken from the compile-time capacity (NO-GO), a dead liveness control (NO-DECISION), a served mode that
    differs (NO-DECISION), a graft that never logs, a wrong binary or kernel, no scratch setting.

    py -3.11 -B -m unittest test_k64j_probe      (from this directory)
"""

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
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
QWEN = HERE.parent / 'sdpa_decode_qwen'
SLICE = HERE.parent / 'sdpa_decode_slice'
for _path in (str(HERE), str(QWEN), str(CI)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import split_model as model  # noqa: E402
import probe_k64j_card_b as probe  # noqa: E402
import test_sdpa_decode_qwen_card_m as card  # noqa: E402

RUNNER = HERE / 'run_card_b.sh'
FIXTURES = HERE / 'fixtures'
RT_ARGS = FIXTURES / 'rt_args_common.1b52c60d.hpp'
DEVICE_OP = FIXTURES / 'sdpa_decode_device_operation.ba2b3fc9.hpp'
ARM = CI / 'lever_n_m3native_run_arm.sh'
GATE = ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
PROFILES = CI / 'qwen_c2_profiles.json'
CARD_B = 'blackhole-F36F768B9A5CAFA0'
CARD_M = 'blackhole-CEF5729692C19E6D'
NL = chr(10)
SERVED_CAPACITY = 131328
CORE_COUNTS = (1, 2, 5, 13, 16, 32)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


# ---------------------------------------------------------------------------------------------
# The split.
# ---------------------------------------------------------------------------------------------

class SplitModelTests(unittest.TestCase):
    def test_the_fixtures_are_the_served_sources(self):
        self.assertEqual(sha(RT_ARGS.read_bytes()), probe.STOCK_KERNELS['rt_args_common.hpp'])
        self.assertTrue(sha(DEVICE_OP.read_bytes()).startswith('ba2b3fc9'))
        text = read(RT_ARGS)
        for anchor in ('valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);',
                       'if constexpr (Sk_chunk_t == 0) {', 'return nearest_pow_of_2_up_to_8<max_size>(seq_len_in_tiles);',
                       'return Sk_chunk_t;', 'if (num_cores_per_batch > int(num_chunks_value)) {'):
            self.assertEqual(text.count(anchor), 1, anchor)
        self.assertIn('inline TreeReductionParams get_tree_reduction_params(uint32_t core_id, uint32_t N) {', read(DEVICE_OP))

    def test_the_fixed_chunk_never_reads_max_dynamic_chunk_size(self):
        """k_chunk_size 256 (every replay config) makes Sk_chunk_t 8, and get_dynamic_Sk_chunk_t returns it under
        `if constexpr`: the factory's max_dynamic_chunk_size (8 by default, 4 with fp32 accumulation) is dead."""
        self.assertEqual((model.max_dynamic_chunk_size(False), model.max_dynamic_chunk_size(True)), (8, 4))
        for cur_pos in (0, 31, 255, 256, 2303, 131327):
            for maximum in (4, 8):
                self.assertEqual(model.dynamic_chunk_tiles(model.SK_CHUNK_T, maximum, cur_pos), 8)
        # The native decode (k_chunk_size 0) takes the power of two of the tiles seen, capped: 256 keys from 225 on.
        self.assertEqual([model.dynamic_chunk_tiles(0, 8, p) for p in (0, 31, 32, 95, 96, 224, 5000)], [1, 1, 2, 4, 4, 8, 8])
        self.assertEqual(model.dynamic_chunk_tiles(0, 4, 5000), 4)

    def test_every_family_of_the_served_geometry_splits_as_its_compile_time_call(self):
        families = model.families(SERVED_CAPACITY)
        self.assertEqual((len(families), families[0], families[-1]), (513, 256, SERVED_CAPACITY))
        for cores in CORE_COUNTS:
            for extent in families:
                for cur_pos in (extent - 256, extent - 255, extent - 129, extent - 2, extent - 1):
                    self.assertTrue(model.same_split(cur_pos, extent, cores), (cores, extent, cur_pos))
                if extent < SERVED_CAPACITY:
                    # One key into the next family is the next family's split, never this one's.
                    self.assertFalse(model.same_split(extent, extent, cores))
                    self.assertTrue(model.same_split(extent, extent + 256, cores))

    def test_the_split_is_the_chunk_count_over_the_cores_reversed(self):
        live = model.split(2304 - 1, 16)
        self.assertEqual(live['num_chunks'], 9)
        self.assertEqual(live['ranges'][:9], tuple((8 - core, 9 - core) for core in range(9)))
        self.assertTrue(all(start == end for start, end in live['ranges'][9:]))
        top = model.split(SERVED_CAPACITY - 1, 16)
        self.assertEqual(top['num_chunks'], 513)
        self.assertEqual(sorted(end - start for start, end in top['ranges']), [32] * 15 + [33])
        self.assertEqual(top['ranges'][15], (0, 33))                # the residual goes to the last core
        self.assertEqual(top['ranges'][0], (481, 513))              # core 0 (the reducer) holds the final chunk

    def test_the_cores_per_head(self):
        self.assertEqual([model.cores_per_head(batch) for batch in (1, 2, 3, 4, 8)], [16, 16, 16, 13, 6])
        with self.assertRaises(ValueError):
            model.cores_per_head(0)

    def test_the_tree(self):
        tree = model.tree_params(0, 16)
        self.assertEqual((tree['is_root'], tree['num_rounds'], tree['children'][:4]), (True, 4, [1, 2, 4, 8]))
        leaf = model.tree_params(1, 16)
        self.assertEqual((leaf['parent'], leaf['send_at_round']), (0, 0))
        # 13 cores (B = 4): vid 12 has no natural child; the root collects the orphans 1 (round 2) and 5 (round 3).
        root = model.tree_params(0, 13)
        self.assertEqual((root['num_rounds'], root['children']), (4, [None, None, 1, 5, None, None]))
        self.assertEqual((model.tree_params(2, 13)['parent'], model.tree_params(1, 13)['children'][0]), (1, 2))
        self.assertEqual(model.tree_params(0, 1), dict(is_root=True, parent=None, send_at_round=None,
                                                       children=[None] * 6, num_rounds=0))

    def test_a_stale_writer_blocks_below_one_chunk_per_core_and_on_every_skip(self):
        for cores in (13, 16):
            blocked = [extent for extent in model.families(SERVED_CAPACITY)
                       if model.stale_writer_hangs(extent - 1, SERVED_CAPACITY, cores)]
            self.assertEqual(blocked, list(range(256, 256 * cores, 256)), cores)
            self.assertTrue(model.stale_writer_hangs(model.UINT32_MAX, SERVED_CAPACITY, cores))
        self.assertEqual(model.stale_writer_hangs(SERVED_CAPACITY - 1, SERVED_CAPACITY, 16), [])
        why = model.stale_writer_hangs(2303, SERVED_CAPACITY, 16)
        self.assertIn((9, 'waits for compute output (no chunk at the runtime extent)'), why)
        self.assertIn((8, 'waits for child 9 (no chunk at the runtime extent)'), why)
        self.assertEqual(len(why), 14)                  # cores 9-15 and the parents 8, 10, 12, 14 of their subtrees
        self.assertFalse([core for core, _why in why if core < 8])        # the root's own children all have a chunk

    def test_the_extent_and_its_runtime_position(self):
        for start in (0, 7, 240, 255, 256, 130816, 131071):
            self.assertEqual(model.extent(start), (start // 256 + 1) * 256)
            self.assertEqual(model.extent_cur_pos(start), start | 255)
            self.assertTrue(model.same_split(model.extent_cur_pos(start), model.extent(start), 16))
        with self.assertRaises(ValueError):
            model.compile_time_cur_pos(100)


def find_gxx():
    for candidate in (os.environ.get('QWEN_PF_GXX'), shutil.which('g++'), shutil.which('clang++')):
        if candidate and Path(candidate).is_file() and 'arm-none-eabi' not in candidate:
            return candidate
    return None


CPP_MAIN = r'''
#include <cstdio>
int main() {
    const int cores_list[] = {CORES};
    const int positions[] = {POSITIONS};
    for (int cores : cores_list) {
        for (int cur_pos : positions) {
            for (int core = 0; core < cores; ++core) {
                auto [pst, n, s, e, wu, wc] = get_workload_for_core(cur_pos, 0, core, cores, 256);
                std::printf("W %d %d %d %u %u %u %u\n", cores, cur_pos, core, pst, n, s, e);
            }
            std::printf("D %d %u %u %u\n", cur_pos, get_dynamic_Sk_chunk_t<0, 8>(cur_pos),
                        get_dynamic_Sk_chunk_t<0, 4>(cur_pos), get_dynamic_Sk_chunk_t<8, 8>(cur_pos));
        }
    }
    for (unsigned n = 1; n <= 40; ++n) {
        for (unsigned core = 0; core < n; ++core) {
            TreeReductionParams p = get_tree_reduction_params(core, n);
            std::printf("T %u %u %d %u %u", n, core, p.is_root ? 1 : 0, p.parent_core_in_group, p.send_at_round);
            for (unsigned r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) std::printf(" %u", p.children_per_round[r]);
            std::printf("\n");
        }
    }
    return 0;
}
'''


class CompiledMirrorTests(unittest.TestCase):
    """The mirror against the C++ itself: the vendored rt_args_common.hpp and get_tree_reduction_params, compiled
    with a host g++ (CI's ubuntu runner has one; locally set QWEN_PF_GXX or skip)."""

    def setUp(self):
        self.gxx = find_gxx()
        if self.gxx is None:
            self.skipTest('no host g++ (set QWEN_PF_GXX)')

    def test_split_model_is_the_compiled_kernel_code(self):
        positions = sorted(set(list(range(0, 600)) + [value for extent in model.families(SERVED_CAPACITY)
                                                      for value in (extent - 256, extent - 1)]))
        cores = (1, 2, 13, 16, 32)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / 'tt-metalium').mkdir()
            (work / 'tt-metalium' / 'constants.hpp').write_text(
                '#pragma once\n#include <cstdint>\nnamespace tt { namespace constants {\n'
                'constexpr uint32_t TILE_HEIGHT = 32;\nconstexpr uint32_t TILE_WIDTH = 32;\n} }\n')
            tree = read(DEVICE_OP)
            start = tree.index('constexpr uint32_t MAX_TREE_REDUCTION_ROUNDS')
            end = tree.index('struct SdpaDecodeDeviceOperation')
            source = ('#include <algorithm>\n#include <cstdint>\n#include "%s"\n%s\n%s'
                      % (RT_ARGS.as_posix(), tree[start:end],
                         CPP_MAIN.replace('CORES', ', '.join(map(str, cores)))
                         .replace('POSITIONS', ', '.join(map(str, positions)))))
            (work / 'mirror.cpp').write_text(source)
            binary = work / 'mirror.exe'
            built = subprocess.run([self.gxx, '-std=c++17', '-O1', '-I', str(work), str(work / 'mirror.cpp'), '-o',
                                    str(binary)], capture_output=True, text=True, timeout=300)
            self.assertEqual(built.returncode, 0, built.stderr[-3000:])
            output = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300).stdout
        checked = {'W': 0, 'D': 0, 'T': 0}
        for line in output.splitlines():
            fields = line.split()
            kind, values = fields[0], [int(value) for value in fields[1:]]
            checked[kind] += 1
            if kind == 'W':
                count, cur_pos, core, pst, chunks, start, stop = values
                self.assertEqual(model.workload(cur_pos, core, count, 256)[:4], (pst, chunks, start, stop), line)
            elif kind == 'D':
                cur_pos, dyn8, dyn4, fixed = values
                self.assertEqual((model.dynamic_chunk_tiles(0, 8, cur_pos), model.dynamic_chunk_tiles(0, 4, cur_pos),
                                  model.dynamic_chunk_tiles(8, 8, cur_pos)), (dyn8, dyn4, fixed), line)
            else:
                count, core, root, parent, send = values[:5]
                children = [None if value == model.UINT32_MAX else value for value in values[5:]]
                params = model.tree_params(core, count)
                self.assertEqual((params['is_root'], params['parent'], params['send_at_round'], params['children']),
                                 (bool(root), None if parent == model.UINT32_MAX else parent,
                                  None if send == model.UINT32_MAX else send, children), line)
        self.assertEqual(checked['D'], len(cores) * len(positions))
        self.assertEqual(checked['W'], sum(cores) * len(positions))
        self.assertEqual(checked['T'], sum(range(1, 41)))


# ---------------------------------------------------------------------------------------------
# The flag bit.
# ---------------------------------------------------------------------------------------------

class FlagBitTests(unittest.TestCase):
    def factory_flags(self, path):
        values = {}
        for name, value in re.findall(r'constexpr uint32_t (kQwen[A-Za-z]+) = 0x([0-9A-Fa-f]+)u;', read(path)):
            if name != 'kQwenSliceAbiTag':
                values[name] = int(value, 16)
        return values

    def test_0x20_is_free_everywhere_and_0x10_is_the_unknown_flag_control(self):
        flags = {}
        flags.update(self.factory_flags(QWEN / 'apply_factory_qwen.py'))
        flags.update(self.factory_flags(SLICE / 'apply_factory_slice.py'))
        self.assertEqual(sorted(flags.values()), [0x1, 0x2, 0x4, 0x8])
        import pooled_attention_replay as pooled
        served = {pooled.QWEN_MASK_TAIL, pooled.QWEN_KV_SHARE, pooled.QWEN_Q_SLICE, pooled.QWEN_KV_READAHEAD}
        self.assertEqual(served, {0x1, 0x2, 0x4, 0x8})
        gate = re.search(r"^SDPA_MODE_FLAGS = (\{[^}]*\})", read(CI / 'lever_n_m3native_gate.py'), flags=re.M).group(1)
        self.assertEqual(sorted(eval(gate).values()), [0x1, 0x2, 0x4, 0x8])  # noqa: S307 - a literal dict
        self.assertEqual(card.UNKNOWN_FLAG & 0xFF, 0x10)
        self.assertIn("('unknown flag 0x10', 8, 2, 0x11,", read(SLICE / 'sdpa_decode_slice_card_b.py'))
        self.assertEqual(probe.K64J_FLAG, 0x20)
        self.assertNotIn(probe.K64J_FLAG, set(flags.values()) | served | {0x10})
        # Every factory refuses what it does not know: the probe's N control expects exactly that text.
        for path in (QWEN / 'apply_factory_qwen.py', SLICE / 'apply_factory_slice.py'):
            self.assertIn('"[QWEN-SDPA] unknown flags {:#x}"', read(path))
        self.assertIn(probe.CAUSAL_NEEDLE, read(QWEN / 'apply_factory_qwen.py'))


# ---------------------------------------------------------------------------------------------
# The probe's helpers.
# ---------------------------------------------------------------------------------------------

def generate_mask_columns(cur_pos, sk_chunk_t=8):
    """dataflow_common.hpp:215-305, one row of the final chunk's mask: True where the causal writer writes -inf."""
    cur_pos_in_chunk = cur_pos % (sk_chunk_t * 32)
    tile, column = cur_pos_in_chunk // 32, cur_pos_in_chunk % 32
    out = []
    for index in range(sk_chunk_t):
        for col in range(32):
            if index < tile:
                out.append(False)
            elif index == tile:
                out.append(col > column)                      # fill_tile_partial: -inf past cur_pos_in_tile
            else:
                out.append(True)
    return out


class HelperTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_the_position_plan_mixes_families_in_every_call(self):
        pairs = probe.position_pairs(probe.EXTENTS, probe.STARTS)
        self.assertEqual(len(pairs), 30)
        self.assertTrue(all(extent - 256 <= p < extent and model.extent(p) == extent for extent, _s, p in pairs))
        calls = probe.group_calls(pairs, 3)
        self.assertEqual(len(calls), 10)
        self.assertEqual({pair for call in calls for pair in call}, set(pairs))
        self.assertTrue(all(len({extent for extent, _s, _p in call}) == 3 for call in calls))
        self.assertEqual(len(probe.group_calls(pairs[:4], 3)), 2)           # the last call wraps to stay full

    def test_the_trace_plan_spans_more_than_fifty_families(self):
        plan = probe.trace_positions(SERVED_CAPACITY, 64, 3, 0)
        self.assertEqual(len(plan), 64)
        self.assertGreater(probe.distinct_families(plan), 50)
        self.assertTrue(all(0 <= p < SERVED_CAPACITY for values in plan for p in values))
        self.assertEqual(model.extent(plan[0][0]), 256)
        self.assertEqual(model.extent(plan[-1][0]), SERVED_CAPACITY)
        self.assertEqual(plan, probe.trace_positions(SERVED_CAPACITY, 64, 3, 0))      # reproducible
        self.assertEqual(probe.skip_positions((5, 6, 7), (0, 2)), (-1, 6, -1))

    def test_the_poisoned_row_keeps_the_extent_and_poisons_the_rest(self):
        torch = self.torch
        table = torch.randperm(2052).to(torch.int32)
        poison = list(range(2052, 2116))
        row = probe.poisoned_row(torch, table, 2304, SERVED_CAPACITY, poison)
        self.assertEqual(row.shape, (2052,))
        self.assertTrue(torch.equal(row[:36], table[:36]))
        self.assertEqual(set(row[36:].tolist()), set(poison))
        self.assertTrue(torch.equal(probe.poisoned_row(torch, table, SERVED_CAPACITY, SERVED_CAPACITY, poison), table))

    def test_the_masks_are_the_causal_writers(self):
        torch = self.torch
        for extent in (512, 2304):
            for start in (0, 7, 127, 240, 255):
                p = extent - 256 + start
                wide = probe.position_mask(torch, [p, p], [extent, extent], 2, extent)
                narrow = probe.position_mask(torch, [p], [extent], 1, 256)
                self.assertEqual(tuple(wide.shape), (2, 1, 24, extent))
                self.assertEqual(tuple(narrow.shape), (1, 1, 12, 256))
                expected = generate_mask_columns(p)
                for mask in (wide[0, 0], wide[1, 0], narrow[0, 0]):
                    tail = mask[:, -256:]
                    self.assertEqual([bool(value) for value in torch.isinf(tail[0])], expected)
                    self.assertTrue(torch.equal(tail, tail[0:1].expand_as(tail)))              # every row sits at p
                    self.assertTrue(bool((mask[:, :-256] == 0).all()))
                    masked = tail[torch.isinf(tail)]
                    self.assertTrue(bool((masked < 0).all()))
                    if masked.numel():
                        self.assertEqual(int(card.int16_view(torch, masked)[0]) & 0xFFFF, 0xFF80)   # generate_mask's bits
        zero = probe.position_mask(torch, [511], [512], 4, 512, zero=True)
        self.assertTrue(bool((zero == 0).all()))
        self.assertEqual(generate_mask_columns(511), [False] * 256)          # E - 1 masks nothing

    def test_the_query_fold(self):
        torch = self.torch
        query = probe.rows_query(torch, 3, 2, 0, 'normal')
        self.assertEqual((tuple(query.shape), query.dtype), ((1, 3, 24, 256), torch.bfloat16))
        self.assertTrue(torch.equal(query, probe.rows_query(torch, 3, 2, 0, 'normal')))
        self.assertFalse(torch.equal(query, probe.rows_query(torch, 3, 2, 0, 'normal', salt=1)))
        generator = torch.Generator().manual_seed(7000)
        tokens = torch.randn(3, 2, 12, 256, generator=generator).to(torch.bfloat16)
        # KV head major: rows 0-5 token 0 of KV head 0, rows 6-11 token 1 of KV head 0, rows 12-17 token 0 of head 1.
        self.assertTrue(torch.equal(query[0, 1, 6:12], tokens[1, 1, 0:6]))
        self.assertTrue(torch.equal(query[0, 1, 12:18], tokens[1, 0, 6:12]))
        zero = probe.rows_query(torch, 1, 1, 0, 'zeroq')
        self.assertTrue(bool((zero[0, 0, ::4] == 0).all()))
        keys = torch.randn(40, 2, 64, 256).to(torch.bfloat16)
        tables = [torch.arange(40, dtype=torch.int32)]
        peaky = probe.rows_query(torch, 1, 1, 0, 'peaky', keys, tables, [100])
        self.assertEqual(tuple(peaky.shape), (1, 1, 12, 256))
        with self.assertRaises(ValueError):
            probe.rows_query(torch, 1, 1, 0, 'peaky')
        with self.assertRaises(ValueError):
            probe.rows_query(torch, 1, 1, 0, 'odd')

    def test_the_served_modes_per_stage_and_the_slice_rule(self):
        import pooled_attention_replay as pooled
        for rows in range(1, 9):
            self.assertEqual(probe.q_slice_saves(rows), pooled.q_slice_saves(rows), rows)
        self.assertEqual(probe.served_flags(0, 8, 2), [])
        self.assertEqual(probe.served_flags(1, 8, 2), [0x1])
        self.assertEqual(probe.served_flags(3, 8, 2), [0x1, 0x3])
        self.assertEqual(probe.served_flags(4, 8, 2), [0x1, 0x3, 0x7])
        self.assertEqual(probe.served_flags(4, 4, 3), [0x1, 0x3])
        self.assertEqual(probe.served_flags(4, 2, 1), [0x1])
        self.assertEqual(probe.binary_stage(dict(flags=True, share=True, stage1=False), True), 4)
        self.assertEqual(probe.binary_stage(dict(flags=True, share=True, stage1=False), False), 3)
        self.assertEqual(probe.binary_stage(dict(flags=False, share=False, stage1=False), False), 0)
        with self.assertRaises(RuntimeError):
            probe.binary_stage(dict(flags=True, share=False, stage1=True), True)

    def test_the_program_keys(self):
        lines = card.factory_lines('x [QWEN-SDPA] flags=0x3 B=3 PNHt=1 St=72 mask_width_t=72 kv_share=true '
                                   'scratch_slots=4 cb_bytes=5\n')
        self.assertEqual(probe.missing_programs(lines, {probe.program_key(2304, 3, 2304, 0x3)}), [])
        self.assertEqual(probe.missing_programs(lines, {probe.program_key(2304, 3, 2304, 0x1)}), [(0x1, 3, 72, 72)])

    def entry(self, kind, differing=0, decisive=True):
        return probe.comparison('S', kind, 'x', differing, decisive)

    def test_the_decision(self):
        live = [dict(label='a', live=True)]
        go = dict(comparisons=[self.entry('split_vs_legacy'), self.entry('half_tile_vs_legacy', 5, False)],
                  liveness=live, failures=[])
        self.assertEqual(probe.decide(go)['verdict'], 'GO')
        nogo = dict(go, comparisons=go['comparisons'] + [self.entry('trace_vs_eager', 3)])
        decision = probe.decide(nogo)
        self.assertEqual((decision['verdict'], decision['decisive_differing']), ('NO-GO', 1))
        self.assertEqual(probe.decide(dict(nogo, failures=['x']))['verdict'], 'NO-DECISION')
        dead = probe.decide(dict(go, liveness=live + [dict(label='b', live=False)]))
        self.assertEqual(dead['verdict'], 'NO-DECISION')
        self.assertIn('liveness controls did not move (b)', dead['reasons'][0])
        empty = probe.decide(dict(comparisons=[self.entry('half_tile_vs_legacy', 0, False)], liveness=[], failures=[]))
        self.assertEqual(empty['verdict'], 'NO-DECISION')
        self.assertEqual(probe.decide(dict(go, error='boom'))['verdict'], 'NO-DECISION')

    def test_the_verdict_line(self):
        report = dict(comparisons=[self.entry('split_vs_legacy'), self.entry('split_vs_legacy', 2),
                                   self.entry('served_vs_legacy', 0, False)],
                      liveness=[dict(label='a', live=True)], failures=[], flag_0x20='unknown',
                      trace_families_distinct=0, binary=dict(stage=4), skip_written='unwritten')
        report['decision'] = probe.decide(report)
        line = probe.verdict_line(report)
        self.assertTrue(line.startswith('K64J_P0 verdict=NO-GO split=1/2 extent=none trace=none skip=none '
                                        'served=1/1 live=1/1 half_tile=none dynamic_chunk=none '
                                        'skipped_rows=unwritten flag_0x20=unknown families=0 binary_stage=4 '
                                        'first_differing=["x"]'), line)

    def test_the_predictions(self):
        rows = probe.split_predictions((2304, SERVED_CAPACITY), SERVED_CAPACITY, (3, 2, 4))
        by = {(row['extent'], row['batch']): row for row in rows}
        self.assertEqual(by[(2304, 3)], dict(extent=2304, batch=3, cores_per_head=16, chunks=9, busiest_core_chunks=1,
                                             same_split_as_capacity_call=True, stale_writer_blocks=14))
        self.assertEqual(by[(SERVED_CAPACITY, 4)]['cores_per_head'], 13)
        self.assertEqual(by[(SERVED_CAPACITY, 2)]['stale_writer_blocks'], 0)

    def test_the_arguments(self):
        args = probe.parse_args(['--out', 'x.json'])
        self.assertEqual((args.capacity, args.extents, args.starts, args.seeds, args.rows, args.batch, args.sections),
                         (131328, list(probe.EXTENTS), list(probe.STARTS), [0, 1, 2], [2, 1], 3, list(probe.SECTIONS)))
        self.assertEqual(args.trace_references, 'slot0')
        for bad in (['--extents', '2300'], ['--extents', '262144'], ['--starts', '256'], ['--rows', '3'],
                    ['--batch', '5'], ['--sections', 'S,Q'], ['--shapes', 'G2B9'], ['--variants', 'odd'],
                    ['--expect-binary-sha256', 'abc'], ['--capacity', '1000'], ['--extents', '2304,2304']):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                probe.parse_args(['--out', 'x.json'] + bad)


# ---------------------------------------------------------------------------------------------
# The contract with the gate, the arm and the profile.
# ---------------------------------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    def test_the_runner_defaults_are_the_c2_arms(self):
        runner = read(RUNNER)
        gate = read(GATE)
        image = re.search(r'^\s*\*-v231\|[^)]*\*-v235\|[^)]*\) image=(sha256:[0-9a-f]{64}) ;;', gate, flags=re.M).group(1)
        self.assertIn('IMAGE=${IMAGE:-%s}' % image, runner)
        c2 = re.search(r'^\s*\*-v235\|\*-v238\) export (.*) ;;$', gate, flags=re.M).group(1)
        self.assertIn('KOPGRAFT64=/home/thatch/opgraft-K64i', c2)
        self.assertIn('M3NATIVE_SDPA_MODES=tail,share,slice', c2)
        self.assertIn('M3NATIVE_REPLAY_GROUP_ROWS=8', c2)
        self.assertIn('G=${KOPGRAFT64:-$HOME/opgraft-K64i}', runner)
        self.assertIn('K64I_SHA256=%s' % probe.K64I_TTNNCPP_SHA256, runner)
        self.assertIn('cf54d716', gate)
        self.assertIn('-e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', read(ARM))

    def test_the_geometry_is_the_served_profiles(self):
        profiles = json.loads(read(PROFILES))
        text = json.dumps(profiles)
        self.assertIn('131328', text)
        self.assertEqual(probe.CAPACITY, 131328)
        self.assertEqual(probe.CAPACITY // card.PAGE, 2052)                       # the reviewed page width
        self.assertEqual(max(probe.EXTENTS), probe.CAPACITY)
        self.assertEqual(probe.EXTENTS, (2304, 16896, 33024, 65792, 98560, 131328))    # plan 2.3 K1's six

    def test_the_served_kernel_hashes_agree_with_the_graft_builders(self):
        runner = read(RUNNER)
        build = read(QWEN / 'build_k64f.sh')
        for name, value in (('READER_ALL', 'dataflow/reader_decode_all.cpp'), ('COMPUTE_ALL', 'compute/sdpa_flash_decode.cpp'),
                            ('WRITER_ALL', 'dataflow/writer_decode_all.cpp'), ('DATAFLOW_COMMON', 'dataflow/dataflow_common.hpp')):
            self.assertIn('%s=%s' % (name, probe.STOCK_KERNELS[value]), build)
        slice_runner = read(SLICE / 'run_card_b.sh')
        for name in ('READER_QWEN', 'COMPUTE_QWEN', 'READER_SLICE', 'WRITER_SLICE'):
            value = re.search(r'^%s=([0-9a-f]{64})$' % name, slice_runner, flags=re.M).group(1)
            self.assertIn('%s=%s' % (name, value), runner)

    def test_the_cpu_suite_runs_this_directory(self):
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/k64j_probe -p 'test_*.py'",
                      read(CPU_WORKFLOW))


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------

def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


BASH = find_bash()
SCRUB = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'IMAGE', 'RESULTS', 'WATCHER', 'WATCHDOG_S', 'KOPGRAFT64',
         'K64J_DRY_RUN', 'EXPECT_TTNNCPP_SHA256', 'CARD_B_ARGS', 'MSYS')
QWEN_KERNELS = {
    'dataflow/reader_decode_qwen.cpp': QWEN / 'stage3' / 'reader_decode_qwen.cpp',
    'compute/sdpa_flash_decode_qwen.cpp': QWEN / 'stage3' / 'sdpa_flash_decode_qwen.cpp',
    'dataflow/reader_decode_qwen_slice.cpp': SLICE / 'reader_decode_qwen_slice.cpp',
    'dataflow/writer_decode_qwen_slice.cpp': SLICE / 'writer_decode_qwen_slice.cpp',
}


def make_graft(root, binary=b'fake _ttnncpp.so'):
    graft = Path(root) / 'graft'
    kernels = graft / 'sdpa_decode' / 'device' / 'kernels'
    for name in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa'):
        (graft / name).mkdir(parents=True)
        (graft / name / 'placeholder.txt').write_bytes(name.encode())
    (graft / '_ttnn.so').write_bytes(b'fake _ttnn.so')
    (graft / '_ttnncpp.so').write_bytes(binary)
    for name, source in QWEN_KERNELS.items():
        (kernels / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, kernels / name)
    write_manifest(graft)
    return graft


def write_manifest(graft):
    files = sorted('./' + path.relative_to(graft).as_posix() for path in graft.rglob('*')
                   if path.is_file() and path.name != 'MANIFEST.sha256')
    (graft / 'MANIFEST.sha256').write_bytes(''.join('%s  %s\n' % (sha((graft / name).read_bytes()), name)
                                                    for name in files).encode())


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, **env):
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_DRY_RUN='1')
        environ.update(env)
        return subprocess.run([BASH, RUNNER.as_posix()], env=environ, capture_output=True, text=True,
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

    def arm_graft_mounts(self):
        arm = read(ARM)
        pairs = set(re.findall(r'-v \$KOPGRAFT64/([A-Za-z0-9_.]+):(/opt/[^:" ]+):ro', arm))
        target = re.search(r'^\s*sdpa_pf_target=(\S+)$', arm, flags=re.M).group(1)
        pairs.add(('sdpa', target))
        self.assertEqual(len(pairs), 7)
        return pairs

    def test_the_dry_run_launches_on_card_b_with_the_arms_graft_mounts(self):
        graft = make_graft(self.dir)
        result = self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(b'fake _ttnncpp.so'))
        argv = self.argv(result)
        self.assertEqual(result.stderr, '')
        self.assertIn('### graft %s: _ttnncpp.so %s, 4 served qwen kernels, manifest verified'
                      % (graft.as_posix(), sha(b'fake _ttnncpp.so')[:16]), result.stdout)
        self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
        self.assertEqual(argv.count('--device'), 1)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-card-b')
        self.assertIn('card=%s (card-b)' % CARD_B, result.stdout)
        mounts = self.mounts(argv)
        prefix = graft.as_posix() + '/'
        grafted = {(m['src'][len(prefix):], m['dst']) for m in mounts if m['src'].startswith(prefix)}
        self.assertEqual(grafted, self.arm_graft_mounts())
        self.assertTrue(all(m.get('readonly') for m in mounts if m['src'].startswith(prefix)))
        bench = {m['dst']: m['src'] for m in mounts if m['dst'].startswith('/bench/')}
        self.assertEqual(sorted(bench), ['/bench/probe_k1_card_b.py', '/bench/probe_k64j_card_b.py',
                                         '/bench/split_model.py', '/bench/test_sdpa_decode_qwen_card_m.py'])
        for dst, src in bench.items():
            self.assertTrue(src.endswith(dst[len('/bench/'):]), src)
        cache = [m for m in mounts if m['dst'] == '/kcache']
        self.assertEqual(len(cache), 1)
        self.assertRegex(cache[0]['src'], r'/results/kcache-[0-9]{8}T[0-9]{6}$')
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        for value in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', 'TT_METAL_CACHE=/kcache', 'TT_METAL_HOME=/opt/tt-metal'):
            self.assertIn(value, env)
        self.assertNotIn('TT_METAL_WATCHER=5', env)
        entry = argv.index('--entrypoint')
        self.assertEqual(argv[entry + 1:entry + 4], ['sh', 'sha256:57cb699489436842d7e7bdd5ab917d509b4492fc95f6f083ef7d2865b488c2ef', '-c'])
        inner = argv[entry + 4]
        self.assertIn('exec python3 -B /bench/probe_k64j_card_b.py "$@"', inner)
        for name in probe.STOCK_KERNELS:
            self.assertIn('/sdpa_decode/device/kernels/' + name, inner)
        tail = argv[entry + 5:]
        self.assertEqual(tail[0], 'probe')
        args = probe.parse_args(tail[1:])
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.no_timing, args.extents),
                         (sha(b'fake _ttnncpp.so'), 300.0, False, list(probe.EXTENTS)))
        self.assertRegex(args.out.as_posix(), r'^/results/probe-[0-9]{8}T[0-9]{6}\.json$')

    def test_the_default_expectation_is_k64i_and_the_watcher_pass(self):
        result = self.run_runner(WATCHER='1', CARD_B_ARGS='--sections S,E')
        argv = self.argv(result)
        self.assertIn('does not exist here; the graft was not checked', result.stdout)
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        self.assertIn('TT_METAL_WATCHER=5', env)
        self.assertTrue(any(m['dst'] == '/opt/tt-metal/generated/watcher' for m in self.mounts(argv)))
        args = probe.parse_args(argv[argv.index('probe') + 1:])
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.no_timing, args.extents, args.seeds,
                          args.variants, args.trace_families, args.sections),
                         (probe.K64I_TTNNCPP_SHA256, 120.0, True, [2304, 33024], [0], ['normal'], 8, ['S', 'E']))
        self.assertIn('WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog 120 s, container timeout 2700 s', result.stdout)

    def test_the_images_own_binary(self):
        argv = self.argv(self.run_runner(KOPGRAFT64='none'))
        self.assertFalse([m for m in self.mounts(argv) if '_ttnncpp.so' in m['dst']])
        self.assertEqual(probe.parse_args(argv[argv.index('probe') + 1:]).expect_binary_sha256, '')

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertNotIn('### argv: ', result.stdout)

    def test_the_graft_checks_refuse_anything_but_the_served_graft(self):
        graft = make_graft(self.dir)
        self.refused(self.run_runner(KOPGRAFT64=graft.as_posix()),
                     'refusing: %s/_ttnncpp.so is %s, not cf54d716669be6b7' % (graft.as_posix(), sha(b'fake _ttnncpp.so')[:16]))
        expect = dict(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(b'fake _ttnncpp.so'))
        reader = graft / 'sdpa_decode' / 'device' / 'kernels' / 'dataflow' / 'reader_decode_qwen_slice.cpp'
        reader.write_bytes(reader.read_bytes() + b'// K64j\n')
        self.refused(self.run_runner(**expect), 'refusing: %s/MANIFEST.sha256 does not verify' % graft.as_posix())
        write_manifest(graft)
        self.refused(self.run_runner(**expect), 'refusing: %s is %s, not the served 0f5a019c'
                     % (reader.as_posix(), sha(reader.read_bytes())))
        reader.unlink()
        write_manifest(graft)
        self.assertIn('2 served qwen kernels', self.run_runner(**expect).stdout)       # a K64g graft: stage 3 only
        shutil.rmtree(graft / 'sdpa')
        self.refused(self.run_runner(**expect), 'refusing: %s/sdpa missing' % graft.as_posix())

    def test_the_serving_pair_is_refused(self):
        self.refused(self.run_runner(QUAL_CARD=CARD_M), 'refusing: QUAL_CARD=%s is card M' % CARD_M)

    def test_the_runner_echoes_the_verdict_line_not_the_summary_line(self):
        lines = [line for line in read(RUNNER).splitlines() if line.startswith('echo ') and 'K64J_P0' in line]
        self.assertEqual(len(lines), 1, lines)
        log = self.dir / 'probe.log'
        log.write_bytes(b'x\nK64J_P0 verdict=GO split=10/10\nSDPA_K64J_P0 passed=True\n')
        result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout, '### K64J_P0 verdict=GO split=10/10' + NL, result.stderr)
        log.write_bytes(b'SDPA_K64J_P0 passed=False\n')
        result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout, '### no K64J_P0 line' + NL, result.stderr)

    def test_a_real_run_resolves_the_board_before_anything_else(self):
        if Path('/dev/tenstorrent/by-id', CARD_B).exists():
            self.skipTest('card B is present on this host: the probe would run')
        result = self.run_runner(K64J_DRY_RUN='0')
        self.refused(result, 'refusing: %s (card B, the qualification card) has no device node here' % CARD_B)
        self.assertFalse((self.dir / 'results').exists())


# ---------------------------------------------------------------------------------------------
# The device flow on a fake ttnn.
# ---------------------------------------------------------------------------------------------

ELEMENT = {'bf16': 2, 'bf8': 1, 'int32': 4}
HALF_TILE_DELTA = 2.0 ** -7


class FakeTensor:
    def __init__(self, address, shape, dtype, nbytes):
        self.address, self._shape, self.dtype, self.nbytes = address, tuple(shape), dtype, nbytes
        self.freed = False

    @property
    def shape(self):
        return self._shape

    def buffer_address(self):
        return self.address


class HostTensor:
    def __init__(self, data):
        self.data = data


class Grid:
    x, y = 11, 10


class FakeTtnn:
    """The ttnn surface the probe touches, with the decode SDPA's semantics as the probe reads them: the causal path
    reads each entry's cur_pos from its tensor (at replay time inside a trace) and skips -1; the non-causal path
    takes capacity - 1; the split and the tree are split_model's; every partial is float32 and merged in tree order;
    legacy adds the whole mask, tail (0x1) only its last 256 columns on the final chunk, share (0x2) reads page-table
    row 0 for every entry; the factory logs one F4 line per new qwen program to fd 1 and refuses unknown flags and a
    qwen sentinel on a causal call. DRAM: a size-keyed LIFO free list whose bytes stay stale.

    broken: 'overread' (the causal extent reads one chunk past cur_pos), 'clamp' (a cur_pos on a family boundary is
    taken one lower: the liveness control goes dead), 'skip_bleeds' (a skipped entry zeroes the next one), 'noskip'
    (a skipped entry is computed at capacity - 1 and written), 'stale_trace' (a replay uses the cur_pos of the
    capture), 'split_capacity' (the causal split at capacity - 1, the visibility at cur_pos), 'served_wrong' (qwen
    outputs + 1), 'half_tile' (a causal call with <= 16 rows moves by a bf16 step), 'accept_0x20'."""

    int32, bfloat16, bfloat8_b = 'int32', 'bf16', 'bf8'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'rm', 'tile', 'dram'

    def __init__(self, torch, broken=(), silent=False, reuse=True):
        self.torch, self.broken, self.silent, self.reuse = torch, set(broken), silent, reuse
        self.transformer = self
        self.memory, self.free, self.next = {}, {}, 0x10000
        self.now = 0.0
        self.programs, self.traces, self.capturing = set(), {}, None
        self.calls = 0

    def clock(self):
        return self.now

    def open_device(self, **options):
        self.options = options
        return self

    def close_device(self, device):
        self.closed = True

    def enable_program_cache(self):
        pass

    def compute_with_storage_grid_size(self):
        return Grid()

    def synchronize_device(self, device):
        pass

    def SDPAProgramConfig(self, **options):  # noqa: N802 - mirrors ttnn
        return options

    def allocate(self, shape, dtype):
        nbytes = int(self.torch.Size(shape).numel()) * ELEMENT[dtype]
        pool = self.free.get(nbytes) if self.reuse else None
        if pool:
            address = pool.pop()
        else:
            address, self.next = self.next, self.next + nbytes
        return FakeTensor(address, shape, dtype, nbytes)

    def from_torch(self, host, *, dtype, layout, device=None, memory_config=None):
        if device is None:
            return HostTensor(host.clone())
        tensor = self.allocate(host.shape, dtype)
        self.memory[tensor.address] = host.clone()
        return tensor

    def copy_host_to_device_tensor(self, host, tensor):
        if tuple(host.data.shape) != tensor.shape:
            raise RuntimeError('copy_host_to_device_tensor: shape mismatch')
        self.memory[tensor.address] = host.data.clone()

    def read(self, tensor):
        if tensor.freed:
            raise RuntimeError('read of a freed tensor')
        return self.memory[tensor.address]

    def to_torch(self, tensor):
        return self.read(tensor).clone()

    def deallocate(self, tensor):
        if not tensor.freed:
            tensor.freed = True
            self.free.setdefault(tensor.nbytes, []).append(tensor.address)

    def begin_trace_capture(self, device, cq_id):
        self.capturing = len(self.traces) + 1
        self.traces[self.capturing] = []
        return self.capturing

    def end_trace_capture(self, device, trace, cq_id):
        self.capturing = None

    def execute_trace(self, device, trace, cq_id, blocking):
        for write in self.traces[trace]:
            write(True)
            self.now += 30e-6

    def release_trace(self, device, trace):
        del self.traces[trace]

    # the op ------------------------------------------------------------------------------
    def attend(self, q, keys, values, table_row, cur_pos, chunk_tiles, scale, cores, causal, mask_row, tail,
               capacity):
        """One entry, in the kernel's order: the split and the tree of split_model; per chunk the max, the exp, the
        sum and P @ V; each core's online merge over its own chunks; then the tree merge. Every intermediate is
        rounded to bf16 (the kernel's im format), so a different split or merge order shows in the output."""
        torch = self.torch

        def bf(value):
            return value.to(torch.bfloat16).float()

        def merge(own, other):
            m = torch.maximum(own[0], other[0])
            a = torch.where(torch.isfinite(own[0]), torch.exp(own[0] - m), torch.zeros_like(m))
            b = torch.where(torch.isfinite(other[0]), torch.exp(other[0] - m), torch.zeros_like(m))
            return m, bf(own[1] * a + other[1] * b), bf(own[2] * a[:, None] + other[2] * b[:, None])

        visible_to = cur_pos
        split_pos = cur_pos
        if causal and 'overread' in self.broken:
            visible_to = split_pos = min(capacity - 1, cur_pos + 256)
        if causal and 'split_capacity' in self.broken:
            split_pos = capacity - 1
        plan = model.split(split_pos, cores, chunk_tiles, 8)
        chunk, chunks = plan['chunk'], plan['num_chunks']
        extent = chunks * chunk
        pages = table_row[:-(-extent // card.PAGE)].long()
        positions = torch.arange(extent)
        rows = q.shape[0]
        per_kv = rows // card.KV_HEADS
        out = torch.empty(rows, card.HEAD_DIM)
        for kv in range(card.KV_HEADS):
            lo, hi = kv * per_kv, (kv + 1) * per_kv
            k = keys[pages, kv].reshape(-1, card.HEAD_DIM)[:extent].float()
            v = values[pages, kv].reshape(-1, card.HEAD_DIM)[:extent].float()
            scores = bf((q[lo:hi].float() @ k.T) * scale)
            if causal:
                scores[:, positions > visible_to] = float('-inf')
            elif mask_row is not None:
                bias = mask_row[lo:hi].float()
                if tail:
                    scores[:, extent - 256:] = scores[:, extent - 256:] + bias[:, -256:]
                else:
                    scores = scores + bias[:, :extent]
            blocks = scores.view(hi - lo, chunks, chunk)
            m = blocks.max(dim=2).values
            finite = torch.isfinite(m)
            p = torch.where(finite[..., None], torch.exp(blocks - torch.where(finite, m, torch.zeros_like(m))[..., None]),
                            torch.zeros_like(blocks))
            l = bf(p.sum(dim=2))
            o = bf(torch.einsum('rnc,ncd->rnd', p, v.view(chunks, chunk, card.HEAD_DIM)))
            partial = {}
            for core, (first, last) in enumerate(plan['ranges']):
                if first == last:
                    continue
                own = (m[:, first], l[:, first], o[:, first])
                for index in range(first + 1, last):
                    own = merge(own, (m[:, index], l[:, index], o[:, index]))
                partial[core] = own

            def reduce(core):
                own = partial[core]
                for child in model.tree_params(core, cores)['children']:
                    if child is not None and child < chunks and child in partial:
                        own = merge(own, reduce(child))
                return own

            result = reduce(0)
            out[lo:hi] = result[2] / result[1][:, None]
        return out

    def paged_scaled_dot_product_attention_decode(self, query, k, v, pages, *, is_causal, scale, program_config,
                                                  memory_config, attn_mask=None, cur_pos_tensor=None):
        torch = self.torch
        sentinel = program_config['q_chunk_size']
        k_chunk = program_config['k_chunk_size']
        qwen = (sentinel & 0xFFFFFF00) == card.MAGIC
        flags = sentinel & 0xFF if qwen else 0
        known = 0xF | (0x20 if 'accept_0x20' in self.broken else 0)
        if qwen and flags & ~known:
            raise RuntimeError('TT_FATAL: [QWEN-SDPA] unknown flags %#x' % flags)
        if qwen and (is_causal or cur_pos_tensor is not None):
            raise RuntimeError('TT_FATAL: [QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor')
        if is_causal and (cur_pos_tensor is None or attn_mask is not None):
            raise RuntimeError('TT_FATAL: a causal paged call needs a cur_pos tensor and no mask')
        if not is_causal and not k_chunk:
            raise RuntimeError('TT_FATAL: Must provide k_chunk_size if paged and non-causal!')
        batches, rows = query.shape[1], query.shape[2]
        table = self.read(pages)
        capacity = table.shape[1] * card.PAGE
        if qwen and flags & 0x1 and attn_mask.shape[3] not in (capacity, 256):
            raise RuntimeError('TT_FATAL: [QWEN-SDPA] tail mask must be full width')
        if qwen and not self.silent:
            key = (flags, batches, capacity // 32, attn_mask.shape[3] // 32)
            if key not in self.programs:
                self.programs.add(key)
                os.write(1, ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=%d kv_share=%s '
                             'scratch_slots=4 cb_bytes=7\n' % (flags, batches, -(-rows // 32), capacity // 32, key[3],
                                                                'true' if flags & 0x2 and batches > 1 else 'false')).encode())
        self.calls += 1
        q = self.read(query).float()
        keys, values = self.read(k), self.read(v)
        mask = None if attn_mask is None else self.read(attn_mask)
        share, tail = bool(flags & 0x2) and batches > 1, bool(flags & 0x1)
        cores = model.cores_per_head(batches)
        snapshot = self.read(cur_pos_tensor).clone() if is_causal else None
        output = self.allocate((1, batches, rows, card.HEAD_DIM), 'bf16')
        if output.address not in self.memory or tuple(self.memory[output.address].shape) != output.shape:
            self.memory[output.address] = torch.zeros(output.shape, dtype=torch.bfloat16)

        def write(replay=False):
            data = self.memory[output.address].clone()
            if is_causal:
                current = snapshot if (replay and 'stale_trace' in self.broken) else self.read(cur_pos_tensor)
                positions = [int(value) for value in current.tolist()]
            else:
                positions = [capacity - 1] * batches
            skipped = []
            for entry in range(batches):
                cur_pos = positions[entry]
                if cur_pos == -1:
                    if 'noskip' not in self.broken:
                        skipped.append(entry)
                        continue
                    cur_pos = capacity - 1
                if is_causal and 'clamp' in self.broken and cur_pos % 256 == 0:
                    cur_pos -= 1
                chunk_tiles = k_chunk // 32
                row = table[0 if share else entry]
                result = self.attend(q[0, entry], keys, values, row, cur_pos, chunk_tiles, scale, cores, is_causal,
                                     None if mask is None else mask[entry, 0], tail, capacity)
                if qwen and 'served_wrong' in self.broken:
                    result = result + 1
                if is_causal and rows <= 16 and 'half_tile' in self.broken:
                    result = result + HALF_TILE_DELTA
                data[0, entry] = result.to(torch.bfloat16)
            if 'skip_bleeds' in self.broken:
                for entry in skipped:
                    if entry + 1 < batches and entry + 1 not in skipped:
                        data[0, entry + 1] = 0
            self.memory[output.address] = data

        if self.capturing is not None:
            self.traces[self.capturing].append(write)
        else:
            write()
        self.now += 20e-6 + 1e-9 * capacity
        return output


class DryRunTests(unittest.TestCase):
    """The probe end to end on the fake: a 4,352-key table (17 chunks, more than the 16 cores per head: both
    workload branches), extents 512 / 2,304 / 4,352."""

    ARGS = ['--capacity', '4352', '--extents', '512,2304,4352', '--starts', '7,255', '--seeds', '0',
            '--variants', 'normal', '--rows', '2', '--trace-families', '4', '--no-timing']
    FULL = ['--capacity', '4352', '--extents', '512,2304,4352', '--starts', '7,255', '--seeds', '0',
            '--variants', 'normal,peaky', '--rows', '2,1', '--trace-families', '6', '--no-timing']
    BINARY = b'fake K64i _ttnncpp.so ' + probe.SLICE_BINARY_MARKER

    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.kernels = self.dir / 'kernels'
        self.stock = {}
        for name in probe.STOCK_KERNELS:
            path = self.kernels / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('// stock %s\n' % name).encode())
            self.stock[name] = sha(path.read_bytes())
        self.binary = self.dir / '_ttnncpp.so'
        self.binary.write_bytes(self.BINARY)

    def tearDown(self):
        self.tmp.cleanup()

    def run_probe(self, fake, extra=(), markers=None, scratch='1', name='probe', binary=None, base=None):
        out = self.dir / ('%s.json' % name)
        if binary is not None:
            self.binary.write_bytes(binary)
        markers = dict(flags=True, share=True, stage1=False) if markers is None else markers
        argv = ['--out', str(out), '--kernel-root', str(self.kernels),
                '--expect-binary-sha256', sha(self.binary.read_bytes())]
        environ = {card.SCRATCH_ENV: scratch} if scratch is not None else {}
        with mock.patch.dict(sys.modules, {'ttnn': fake}), \
                mock.patch.object(card, 'loaded_binary', return_value=(str(self.binary), markers)), \
                mock.patch.dict(probe.STOCK_KERNELS, self.stock), \
                mock.patch.dict(os.environ, environ), \
                mock.patch.object(probe, 'clock', fake.clock), \
                mock.patch.object(card, 'WATCHDOG', card.WATCHDOG), mock.patch.object(probe, 'WATCHDOG', probe.WATCHDOG), \
                mock.patch.object(probe.k1, 'WATCHDOG', probe.k1.WATCHDOG), \
                mock.patch.object(probe, 'print', create=True), mock.patch('sys.stdout'):
            if scratch is None:
                os.environ.pop(card.SCRATCH_ENV, None)
            status = probe.main(argv + list(self.ARGS if base is None else base) + list(extra))
        return status, json.loads(out.read_text())

    def kinds(self, report):
        return {kind: (row['equal'], row['runs']) for kind, row in report['tally'].items()}

    def test_go_end_to_end(self):
        fake = FakeTtnn(self.torch)
        status, report = self.run_probe(fake, base=self.FULL)
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict']), (0, True, 'GO'))
        self.assertEqual(report['binary']['stage'], 4)
        self.assertEqual(report['kernels']['stock'], self.stock)
        kinds = self.kinds(report)
        # S: 6 (E, s) pairs in 2 calls of 3, 2 variants: 12 entries per rows value; served 0x1 / 0x3 per entry.
        self.assertEqual(kinds['split_vs_legacy'], (12, 12))
        self.assertEqual(kinds['half_tile_vs_legacy'], (12, 12))
        # E: G4B3 (0x1, 0x3) and G8B2 (0x1, 0x3, 0x7) at 3 extents.
        self.assertEqual(kinds['extent_vs_legacy'], (6, 6))
        self.assertEqual(kinds['served_vs_legacy'], (2 * 24 + 3 * 5, 2 * 24 + 3 * 5))
        self.assertEqual(kinds['trace_vs_eager'], (6, 6))
        self.assertEqual(kinds['trace_vs_legacy'], (6, 6))
        self.assertEqual(kinds['trace_skip_live'], (5, 5))
        self.assertEqual(kinds['skip_live'], (5, 5))
        self.assertEqual(kinds['dynamic_vs_fixed'], (2, 2))
        self.assertTrue(all(entry['decisive'] for entry in report['comparisons']
                            if entry['kind'] in probe.DECISIVE_KINDS))
        # Liveness: S (2 rows x 2 calls, entries below the capacity) and E (2 shapes x 2 extents x entries).
        self.assertTrue(report['liveness'] and all(entry['live'] for entry in report['liveness']))
        self.assertEqual(len(report['liveness']), 2 * 4 + 2 * 3 + 2 * 2)
        self.assertEqual(report['flag_0x20'], 'unknown')
        self.assertTrue(report['refusals']['qwen sentinel on a causal call']['matched'])
        self.assertEqual(report['skip_written'], 'unwritten')
        self.assertEqual(report['skip_idle_call'], 'returned')
        self.assertTrue(all(entry['poison_address_reused'] for entry in report['comparisons'] if entry['kind'] == 'skip_live'))
        self.assertGreaterEqual(report['trace_families_distinct'], 3)
        self.assertEqual(report['requested_programs'], sorted(report['requested_programs']))
        self.assertIn([0x7, 2, 4352 // 32, 4352 // 32], report['requested_programs'])
        self.assertTrue(report['verdict_line'].startswith('K64J_P0 verdict=GO split=12/12 extent=6/6 trace=12/12 '
                                                          'skip=10/10 served=63/63 live=18/18 half_tile=12/12 '
                                                          'dynamic_chunk=2/2 skipped_rows=unwritten flag_0x20=unknown'),
                        report['verdict_line'])
        self.assertEqual(fake.options['trace_region_size'], 16 << 20)
        self.assertNotIn('timing', report)

    def test_an_extent_read_past_its_chunk_is_no_go(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'overread'}))
        self.assertEqual((status, report['failures']), (1, []))
        self.assertEqual(report['decision']['verdict'], 'NO-GO')
        # Only the entries at p = C - 1 cannot read further (the table ends there).
        differing = [entry for entry in report['comparisons'] if entry['kind'] == 'split_vs_legacy' and entry['differing']]
        equal = [entry for entry in report['comparisons'] if entry['kind'] == 'split_vs_legacy' and not entry['differing']]
        self.assertTrue(differing)
        self.assertTrue(all(entry['position'] == 4351 for entry in equal), equal)
        self.assertEqual(self.kinds(report)['extent_vs_legacy'], (2, 6))       # E = C again

    def test_a_trace_that_ignores_the_rewritten_position_is_no_go(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'stale_trace'}), ['--sections', 'T'])
        self.assertEqual(report['decision']['verdict'], 'NO-GO')
        equal, runs = self.kinds(report)['trace_vs_eager']
        self.assertEqual((equal, runs), (1, 4))                 # only the capture's own positions replay right

    def test_every_entry_can_be_referenced_in_the_trace(self):
        status, report = self.run_probe(FakeTtnn(self.torch), ['--sections', 'T', '--trace-references', 'all'])
        self.assertEqual((status, report['decision']['verdict']), (0, 'GO'))
        self.assertEqual(self.kinds(report)['trace_vs_legacy'], (12, 12))
        self.assertEqual(self.kinds(report)['trace_vs_eager'], (4, 4))

    def test_a_skip_that_disturbs_a_live_entry_is_no_go(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'skip_bleeds'}), ['--sections', 'K,T'])
        self.assertEqual(report['decision']['verdict'], 'NO-GO')
        self.assertLess(self.kinds(report)['skip_live'][0], 5)

    def test_a_skip_that_writes_is_recorded_not_failed(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'noskip'}), ['--sections', 'K'])
        self.assertEqual((status, report['decision']['verdict']), (0, 'GO'))
        self.assertEqual(report['skip_written'], 'written')

    def test_a_split_from_the_compile_time_capacity_is_no_go(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'split_capacity'}), ['--sections', 'S,E'])
        self.assertEqual(report['decision']['verdict'], 'NO-GO')
        self.assertLess(self.kinds(report)['split_vs_legacy'][0], 12)

    def test_a_dead_liveness_control_decides_nothing(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'clamp'}), ['--sections', 'S,E'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertIn('liveness controls did not move', report['decision']['reasons'][0])
        self.assertEqual(self.kinds(report)['split_vs_legacy'], (6, 6))             # starts 7, 255: no boundary

    def test_a_served_mode_that_differs_fails_the_run(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'served_wrong'}), ['--sections', 'E'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertTrue(all('served mode differs from legacy' in failure for failure in report['failures']))

    def test_the_half_tile_is_recorded_and_the_verdict_stands(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'half_tile'}), ['--sections', 'S', '--rows', '2,1'])
        self.assertEqual((status, report['decision']['verdict']), (0, 'GO'))
        self.assertEqual(self.kinds(report)['half_tile_vs_legacy'], (0, 6))
        self.assertEqual(self.kinds(report)['split_vs_legacy'], (6, 6))
        self.assertIn(' half_tile=0/6 ', report['verdict_line'])

    def test_a_binary_that_accepts_0x20_is_warned(self):
        status, report = self.run_probe(FakeTtnn(self.torch, broken={'accept_0x20'}), ['--sections', 'N,E'])
        self.assertEqual(report['flag_0x20'], 'accepted')
        self.assertTrue(any('ACCEPTS flag 0x20' in warning for warning in report['warnings']))

    def test_a_graft_that_never_logs_was_not_executed(self):
        status, report = self.run_probe(FakeTtnn(self.torch, silent=True), ['--sections', 'E'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertEqual(len(report['failures']), 15)
        self.assertTrue(all('graft mounted, not executed' in failure for failure in report['failures']))

    def test_the_binary_the_kernels_and_the_scratch_are_checked_first(self):
        fake = FakeTtnn(self.torch)
        status, report = self.run_probe(fake, ['--expect-binary-sha256', 'f' * 64], name='sha')
        self.assertEqual(status, 1)
        self.assertIn('not the expected ffffffffffffffff', report['failures'][0])
        self.assertEqual((report['comparisons'], fake.calls), ([], 0))
        stock = self.kernels / 'rt_args_common.hpp'
        stock.write_bytes(b'// changed\n')
        status, report = self.run_probe(FakeTtnn(self.torch), name='kernel')
        self.assertIn('not the ones this probe reads', report['failures'][0])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        stock.write_bytes(('// stock %s\n' % 'rt_args_common.hpp').encode())
        status, report = self.run_probe(FakeTtnn(self.torch), scratch=None, name='scratch')
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required', report['failures'][0])

    def test_a_stock_binary_runs_legacy_only(self):
        status, report = self.run_probe(FakeTtnn(self.torch), ['--sections', 'S,E,N'], binary=b'stock',
                                        markers=dict(flags=False, share=False, stage1=False))
        self.assertEqual((status, report['binary']['stage'], report['decision']['verdict']), (0, 0, 'GO'))
        self.assertNotIn('served_vs_legacy', report['tally'])
        self.assertEqual(report['flag_0x20'], 'n/a(stock)')

    def test_the_timing_is_recorded(self):
        base = [arg for arg in self.ARGS if arg != '--no-timing']
        status, report = self.run_probe(FakeTtnn(self.torch), ['--sections', 'K', '--warmup', '0', '--iters', '2',
                                                               '--rounds', '2'], name='timing', base=base)
        self.assertEqual(status, 0)
        names = [row['name'] for row in report['timing']['rows']]
        self.assertEqual(names, ['runtime E512', 'compile E512', 'runtime E2304', 'compile E2304', 'runtime E4352',
                                 'compile E4352', 'skip0', 'skip1', 'skip2', 'skip3'])
        self.assertEqual(sorted(report['timing']['runtime_over_compile']), ['E2304', 'E4352', 'E512'])
        self.assertTrue(all(row['eager']['n'] == 4 for row in report['timing']['rows']))
        self.assertEqual(sorted(report['timing']['skip_us']), ['skip0', 'skip1', 'skip2', 'skip3'])

if __name__ == '__main__':
    unittest.main()
