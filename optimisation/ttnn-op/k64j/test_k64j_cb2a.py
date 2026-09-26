"""CPU checks for CB2a's sections of the K64j card-B harness (k64j_card_b.py K2, X7 and Z; s2-design.md W10a and
6.2); no device, no ttnn.

  - the helpers: the served masks are the pinned mask kernel (attention_mask_replay.cpp:18-33, its Python twin
    mask_position, card M's build_mask, and attention_head_fold.causal_mask for the causal truth); the narrow mask
    is the last 256 columns of the wide one at every family of the served geometry; the boundary cap; K2's tickets
    are the design's positions and families; the G8B2 layout is LAYOUT(16, 8) and the fold puts each token on the
    row the mask kernel gives its position; the native and the extent split alike from 128 on and not below; the
    token queries;
  - the contract: K2's reference kwargs are the model's native decode call (docker/qwen-c2-graft/graft/attention/
    tp.py, parsed), SerialAttentionReader forwards them in one B = 1 call per row, the subject's are
    attention_parallel's plus cur_pos_tensor (checked on the calls the harness makes); the constants; the runner's
    CB2a watcher pass carries the reduced set and the full pass the design's;
  - the flow on the fake ttnn (test_k64j_card_b.FakeExtentTtnn): PASS end to end with K2's own verdict, and each
    broken variant on the section that must catch it - a K2 reference at the wrong cur_pos (p + 1, p - 1, E - 1),
    a half-tile Q that moves (K2), a mask read at [C - 256, C) (X7), a narrow-mask batch offset that K1's X cannot
    see and X7 and K2 can, a stale writer (Z: the watchdog fires; at 4,352 keys it does not hang); a section that
    raises, the deadline and SIGTERM on the new sections.

    py -3.11 -B -m unittest test_k64j_cb2a      (from this directory; the runner test needs Git Bash on Windows)
"""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_k64j_card_b as base  # noqa: E402 - sys.path (k64j, k64j_probe, sdpa_decode_qwen, scripts/ci), the fake

import attention_batch  # noqa: E402 - scripts/ci
import attention_head_fold  # noqa: E402 - scripts/ci
import attention_mask_replay  # noqa: E402 - scripts/ci (pinned: imported, never edited)

card_b, probe, model, card = base.card_b, base.probe, base.model, base.card
FakeExtentTtnn = base.FakeExtentTtnn
ROOT = base.ROOT
CI = ROOT / 'scripts' / 'ci'
NATIVE = ROOT / card_b.NATIVE_DECODE['source']
GRAFT_SHAS = ROOT / 'docker' / 'qwen-c2-graft' / 'graft.sha256'
PARALLEL = CI / 'attention_parallel.py'
REPLAY = CI / 'attention_replay.py'
SERVED_CAPACITY = 131328
NL = chr(10)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def int16(torch, tensor):
    return card.int16_view(torch, tensor)


# ---------------------------------------------------------------------------------------------
# The helpers.
# ---------------------------------------------------------------------------------------------

class MaskTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_the_row_positions_are_the_pinned_kernels(self):
        for word in (0, 7, 32, 127, 240, 255, 131079):
            self.assertEqual(card_b.mask_row_positions(word),
                             [[attention_mask_replay.mask_position(word, 8, batch, head) for head in range(96)]
                              for batch in range(2)])
            self.assertEqual(card_b.mask_row_positions(word), card.mask_positions(word, card_b.SERVED_OFFSETS, rows=8))
        self.assertEqual(card_b.mask_row_positions(9, rows=4, batches=3, offset=4),
                         [[attention_mask_replay.mask_position(9, 4, batch, head, 4) for head in range(48)]
                          for batch in range(3)])

    def test_the_narrow_mask_is_the_wide_masks_last_chunk_at_every_family(self):
        torch = self.torch
        for extent in model.families(SERVED_CAPACITY):
            for offset in (0, 7, 127, 128, 240, 241, 255):
                start = extent - 256 + offset
                narrow = card_b.narrow_mask(torch, start)
                tail = card_b.served_mask(torch, start, extent, width=256)
                self.assertTrue(torch.equal(int16(torch, narrow), int16(torch, tail)), (extent, offset))
        self.assertEqual(tuple(narrow.shape), (2, 1, 96, 256))
        bits = set(int16(torch, card_b.narrow_mask(torch, 131072 + 7)).unique().tolist())
        self.assertEqual(bits, {0, -128})                                        # 0x0000 and 0xff80 only

    def test_the_wide_mask_is_card_ms_and_the_causal_truth(self):
        torch = self.torch
        for extent in (256, 512, 2304):
            for offset in (0, 7, 127, 240, 255):
                start = extent - 256 + offset
                wide = card_b.wide_mask(torch, start, extent)
                narrow = card_b.narrow_mask(torch, start)
                self.assertEqual(tuple(wide.shape), (2, 1, 96, extent))
                self.assertTrue(torch.equal(int16(torch, wide), int16(torch, card.build_mask(
                    torch, extent, start, card_b.SERVED_OFFSETS, rows=8))))
                self.assertTrue(torch.equal(int16(torch, narrow), int16(torch, card.build_mask(
                    torch, 256, start & 255, card_b.SERVED_OFFSETS, rows=8))))
                self.assertFalse(bool(torch.isinf(wide[..., :extent - 256].float()).any()))   # the zero upload
                if offset + 16 <= 256:                   # a ticket inside its family: the host oracle's causal mask
                    for entry, group in enumerate(card_b.SERVED_OFFSETS):
                        truth = attention_head_fold.causal_mask(8, start + group, extent)[0, 0]
                        self.assertTrue(torch.equal(int16(torch, wide[entry, 0]), int16(torch, truth)), (extent, offset))
                        self.assertTrue(torch.equal(int16(torch, narrow[entry, 0]), int16(torch, truth[:, -256:])))

    def test_the_boundary_cap_and_the_rows_past_e(self):
        torch = self.torch
        for extent in (256, 512, SERVED_CAPACITY):
            first = extent - 256
            self.assertEqual([card_b.accept_limit(first + r) for r in range(240, 256)], list(range(16, 0, -1)))
            self.assertEqual(card_b.accept_limit(first + 7), 16)
            self.assertEqual(card_b.valid_positions(first + 255), [extent - 1])
            self.assertEqual(card_b.valid_positions(first + 7), list(range(first + 7, first + 23)))
            for offset in (241, 250, 255):       # a row at E - 1 or past E sees all of [E - 256, E): no -inf in it
                narrow = card_b.narrow_mask(torch, first + offset)
                positions = card_b.mask_row_positions(offset)
                for entry in range(2):
                    for head in range(96):
                        self.assertEqual(bool(torch.isinf(narrow[entry, 0, head].float()).any()),
                                         positions[entry][head] < 255, (offset, entry, head))

    def test_the_idle_starts(self):
        self.assertEqual(card_b.IDLE_STARTS, (0, 32))
        self.assertEqual(card_b.Z_STARTS[:2], card_b.IDLE_STARTS)
        for start in card_b.IDLE_STARTS:
            self.assertEqual((model.extent(start), start & 255, model.extent(start) - 1), (256, start, 255))


class PlanTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_k2s_tickets_are_the_designs(self):
        tickets = card_b.k2_tickets(card_b.K2_SWEEP, card_b.K2_FLOOR, card_b.CB2_EXTENTS, card_b.CB2_STARTS)
        kinds = [ticket['kind'] for ticket in tickets]
        self.assertEqual((kinds.count('sweep'), kinds.count('family'), kinds.count('floor')), (173, 25, 28))
        self.assertEqual(kinds, sorted(kinds, key=('sweep', 'family', 'floor').index))     # decisive first
        sweep = [ticket for ticket in tickets if ticket['kind'] == 'sweep']
        self.assertEqual([ticket['start'] for ticket in sweep], list(range(128, 301)))
        self.assertEqual({ticket['extent'] for ticket in sweep}, {256, 512})
        covered = {position for ticket in sweep for position in card_b.k2_compared(ticket)}
        self.assertTrue(set(range(128, 301)) <= covered)
        # At E = 256 every row slot of the ticket meets every mask word it can: the cap's 15 short tickets included.
        slots = {(ticket['start'] & 255, position - ticket['start']) for ticket in sweep if ticket['extent'] == 256
                 for position in card_b.k2_compared(ticket)}
        self.assertEqual(slots, {(word, slot) for word in range(128, 256) for slot in range(16) if word + slot < 256})
        self.assertEqual([(ticket['extent'], ticket['start']) for ticket in tickets if ticket['kind'] == 'family'],
                         [(extent, extent - 256 + offset) for extent in (2304, 4352, 16640, 65792, 131328)
                          for offset in (0, 7, 127, 240, 255)])
        for ticket in tickets:
            rows = card_b.k2_compared(ticket)
            self.assertTrue(rows and all(position < model.extent(ticket['start']) for position in rows), ticket)
            if ticket['kind'] != 'floor':
                self.assertGreaterEqual(ticket['start'], card_b.MIN_LIVE_START)
                self.assertEqual(rows, card_b.valid_positions(ticket['start']))
        floor = {position for ticket in tickets if ticket['kind'] == 'floor' for position in card_b.k2_compared(ticket)}
        self.assertEqual(floor, set(range(100, 128)))
        self.assertIsNone(card_b.parse_range('', 'x'))
        self.assertEqual(len(card_b.k2_tickets((128, 130), None, (2304,), (7,))), 4)

    def test_the_floor_is_the_native_chunk(self):
        """design 1.4 #9: the native chunk is 128 keys at 127 and 256 from 128 on, and from 128 on the native B = 1
        call splits every family exactly as the extent call at E - 1 with B = 2 (16 cores per head for both)."""
        self.assertEqual(attention_head_fold.chunk_groups(127, 1)[0]['signature'], (128, 128))
        self.assertEqual(attention_head_fold.chunk_groups(128, 1)[0]['signature'], (256, 256))
        self.assertEqual((model.dynamic_chunk_tiles(0, 8, 127), model.dynamic_chunk_tiles(0, 8, 128)), (4, 8))
        cores = model.cores_per_head(1)
        self.assertEqual((cores, model.cores_per_head(card_b.SERVED_BATCH)), (16, 16))
        for extent in model.families(SERVED_CAPACITY):
            for position in {extent - 256, extent - 129, extent - 1}:
                if position >= card_b.MIN_LIVE_START:
                    self.assertEqual(model.split(position, cores, 0, 8), model.split(extent - 1, cores), position)
        for position in range(100, 128):
            self.assertNotEqual(model.split(position, cores, 0, 8)['chunk'], model.split(255, cores)['chunk'])

    def test_the_layout_is_one_g8b2_bundle(self):
        layout = attention_head_fold.parallel_groups(256, 16, max_group_rows=8)
        self.assertEqual([[(group['offset'], group['rows']) for group in bundle] for bundle in layout],
                         [[(offset, card_b.SERVED_ROWS) for offset in card_b.SERVED_OFFSETS]])
        # The pinned reader captured at any admitted start inside its family would bundle the same (design 2.3).
        for start in list(range(128, 700)) + [16384 + 7, 131072 + 240]:
            if start & 255 > 240:
                continue
            got = attention_head_fold.parallel_groups(start, 16, max_group_rows=8)
            self.assertEqual([[(group['offset'], group['rows']) for group in bundle] for bundle in got],
                             [[(0, 8), (8, 8)]], start)

    def test_the_fold_puts_each_token_on_its_mask_row(self):
        torch = self.torch
        tokens = torch.arange(16 * 12 * 256, dtype=torch.float32).reshape(1, 16, 12, 256)
        folded = card.fold_entries(torch, tokens, card_b.SERVED_OFFSETS, card_b.SERVED_ROWS)
        self.assertEqual(tuple(folded.shape), (1, 2, 96, 256))
        relative = card_b.mask_row_positions(0)
        for entry in range(2):
            for head in range(96):
                self.assertTrue(torch.equal(folded[0, entry, head], tokens[0, relative[entry][head], (head // 48) * 6
                                                                          + head % 6]), (entry, head))
        self.assertTrue(torch.equal(card.unfold_entries(torch, folded, 8), tokens))
        for entry, offset in enumerate(card_b.SERVED_OFFSETS):
            self.assertTrue(torch.equal(folded[:, entry:entry + 1],
                                        attention_head_fold.fold_query(tokens[:, offset:offset + 8])))
        self.assertTrue(torch.equal(card_b.ticket_query(torch, lambda p: tokens[0, p - 500], 500), folded))

    def test_the_token_queries(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(1)
        keys = (torch.randn(80, 2, 64, 256, generator=generator) * 2).to(torch.bfloat16)
        table = torch.randperm(80, generator=generator).to(torch.int32)
        query = card_b.token_query(torch, 0, 'normal', 300)
        self.assertEqual((tuple(query.shape), query.dtype), ((12, 256), torch.bfloat16))
        self.assertTrue(torch.equal(query, card_b.token_query(torch, 0, 'normal', 300)))
        self.assertFalse(torch.equal(query, card_b.token_query(torch, 1, 'normal', 300)))
        self.assertFalse(torch.equal(query, card_b.token_query(torch, 0, 'normal', 301)))
        peaky = card_b.token_query(torch, 0, 'peaky', 300, keys, table)
        self.assertFalse(torch.equal(peaky, query))

        def key(position, kv):
            return keys[int(table[position // 64]), kv, position % 64].float()

        for head in range(12):
            scores = torch.stack([peaky[head].float() @ key(position, head // 6) for position in range(400)])
            top = set(scores.topk(10).indices.tolist())
            self.assertTrue({300, 301} <= top, (head, sorted(top)))              # its own key and the one past it
        for bad in (dict(variant='peaky'), dict(variant='zeroq')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                card_b.token_query(torch, 0, bad['variant'], 300)

    def test_the_native_cur_pos_is_the_rows_own_position(self):
        self.assertEqual([card_b.native_cur_pos(position) for position in (128, 300, 131327)], [128, 300, 131327])


# ---------------------------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------------------------

def function_node(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def calls(node, attribute):
    return [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == attribute]


def assignment(node, name):
    return next(statement.value for statement in ast.walk(node) if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets))


class ContractTests(unittest.TestCase):
    def test_k2s_reference_is_the_models_native_decode_call(self):
        self.assertEqual(sha(NATIVE.read_bytes()), card_b.NATIVE_DECODE['sha256'])
        pinned = dict(reversed(line.split()) for line in GRAFT_SHAS.read_text(encoding='utf-8').splitlines() if line)
        self.assertEqual(pinned['graft/attention/tp.py'], card_b.NATIVE_DECODE['sha256'])
        lines = card_b.NATIVE_DECODE['lines']
        for name, (config_lines, call_lines) in (('_decode_from_prep', (lines['program_config'], lines['call'])),
                                                 ('forward_decode', tuple(lines['forward_decode_same'].split(', ')))):
            with self.subTest(function=name):
                function = function_node(NATIVE, name)
                call = [node for node in calls(function, 'paged_scaled_dot_product_attention_decode')
                        if any(keyword.arg == 'page_table_tensor' for keyword in node.keywords)]
                self.assertEqual(len(call), 1)
                call = call[0]
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                self.assertEqual(sorted(keywords), sorted(card_b.NATIVE_CALL_KWARGS))
                self.assertEqual(len(call.args), 3)                                      # q, keys, values
                for missing in card_b.NATIVE_DECODE['not_passed']:
                    self.assertNotIn(missing, keywords)
                self.assertEqual(ast.unparse(keywords['scale']), 'self.scale')
                self.assertEqual(ast.unparse(keywords['memory_config']), '_L1')
                self.assertEqual(ast.unparse(assignment(function, '_L1')), 'ttnn.L1_MEMORY_CONFIG')
                config = assignment(function, ast.unparse(keywords['program_config']))
                options = {keyword.arg: keyword.value for keyword in config.keywords}
                self.assertEqual({key: ast.literal_eval(value) for key, value in options.items()
                                  if key != 'compute_with_storage_grid_size'}, card_b.NATIVE_PROGRAM_CONFIG)
                self.assertEqual(ast.unparse(options['compute_with_storage_grid_size']), '(_sdpa_grid.x, _sdpa_grid.y)')
                self.assertEqual(ast.unparse(assignment(function, '_sdpa_grid')),
                                 'self.mesh.compute_with_storage_grid_size()')
                self.assertEqual(('%d-%d' % (config.lineno, config.end_lineno), '%d-%d' % (call.lineno, call.end_lineno)),
                                 (config_lines, call_lines))
        tree = ast.parse(NATIVE.read_text(encoding='utf-8'))
        scale = [statement for statement in ast.walk(tree) if isinstance(statement, ast.Assign)
                 and ast.unparse(statement.targets[0]) == 'self.scale']
        self.assertEqual(len(scale), 1)
        scale = scale[0]
        self.assertEqual((ast.unparse(scale.value), str(scale.lineno)), ('self.HD ** (-0.5)', lines['scale']))
        self.assertEqual(card_b.NATIVE_SCALE, 256 ** -0.5)
        self.assertEqual(card_b.NATIVE_SCALE, card.SCALE)
        self.assertEqual(set(card_b.NATIVE_DECODE['program_config']) - {'compute_with_storage_grid_size'},
                         set(card_b.NATIVE_PROGRAM_CONFIG))
        self.assertTrue(card_b.UNVERIFIED and all(isinstance(item, str) for item in card_b.UNVERIFIED))
        printed = card_b.native_decode_lines()
        self.assertEqual(len(printed), 1 + len(card_b.UNVERIFIED))
        self.assertTrue(all(line.startswith('UNVERIFIED K2: ') for line in printed[1:]))
        self.assertIn('k_chunk_size', printed[0])

    def test_the_solo_reader_issues_one_b1_call_per_row_and_forwards_the_kwargs(self):
        class Tensor:
            def __init__(self, shape, name):
                self.shape, self.name = shape, name

        class Operations:
            DRAM_MEMORY_CONFIG = 'dram'

            def __init__(self):
                self.transformer, self.sdpa, self.slices = self, [], []

            def slice(self, tensor, begin, end, memory_config):
                self.slices.append((begin, end, memory_config))
                return Tensor(tuple(e - b for b, e in zip(begin, end)), 'row%d' % begin[1])

            def paged_scaled_dot_product_attention_decode(self, *args, **kwargs):
                self.sdpa.append((args, kwargs))
                return Tensor((1, 1, 12, 256), 'out')

            def deallocate(self, tensor):
                pass

            def concat(self, outputs, dim, memory_config):
                return Tensor((1, len(outputs), 12, 256), 'cat')

        operations = Operations()
        positions, pages = ['pos%d' % index for index in range(4)], ['pages%d' % index for index in range(4)]
        reader = attention_batch.SerialAttentionReader(operations, positions, pages)
        native = dict(scale=0.0625, program_config='native config', memory_config='l1')
        reader(Tensor((1, 4, 12, 256), 'q'), 'K', 'V', page_table_tensor='table', cur_pos_tensor='cur', **native)
        self.assertEqual(len(operations.sdpa), 4)
        for index, (args, kwargs) in enumerate(operations.sdpa):
            self.assertEqual((args[0].shape, args[1:]), ((1, 1, 12, 256), ('K', 'V')))
            self.assertEqual(kwargs, dict(page_table_tensor=pages[index], cur_pos_tensor=positions[index], **native))
            self.assertEqual(operations.slices[index], ((0, index, 0, 0), (1, index + 1, 12, 256), 'dram'))

    def test_the_subjects_kwargs_are_the_replays(self):
        """The extent reader's call is attention_parallel's (plus cur_pos_tensor, design W1 execute_extent) with the
        pooled config (attention_replay.py:53-54: the grid, exp_approx_mode False, k_chunk_size 256)."""
        call = calls(function_node(PARALLEL, 'execute'), 'paged_scaled_dot_product_attention_decode')[0]
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertEqual(sorted(keywords), sorted(SERVED_KWARGS - {'cur_pos_tensor'}))
        self.assertIs(ast.literal_eval(keywords['is_causal']), False)
        config = calls(function_node(REPLAY, '__init__'), 'SDPAProgramConfig')[0]
        options = {keyword.arg: ast.unparse(keyword.value) for keyword in config.keywords}
        self.assertEqual((options['exp_approx_mode'], options['k_chunk_size']), ('False', str(card_b.K_CHUNK)))

    def test_the_constants(self):
        self.assertEqual((card_b.SERVED_FLAGS, card_b.COMPILE_FLAGS), (0x27, 0x7))
        self.assertEqual(card_b.reference_flags(card_b.SERVED_FLAGS), card_b.COMPILE_FLAGS)
        self.assertTrue(card_b.valid_combo('G8B2', card_b.SERVED_FLAGS))
        self.assertEqual(card_b.SHAPES['G8B2'], (card_b.SERVED_ROWS, card_b.SERVED_BATCH))
        hazard = tuple(extent for extent in model.families(SERVED_CAPACITY)
                       if model.stale_writer_hangs(extent - 1, SERVED_CAPACITY, model.cores_per_head(2)))
        self.assertEqual(card_b.Z_FAMILIES, hazard)
        self.assertEqual(len(card_b.Z_FAMILIES), 15)
        self.assertEqual(card_b.Z_STARTS, (0, 32, 7, 127, 240, 255))
        self.assertEqual((card_b.CB2_EXTENTS, card_b.CB2_STARTS), ((2304, 4352, 16640, 65792, 131328),
                                                                   (0, 7, 127, 240, 255)))
        self.assertEqual((card_b.K2_SWEEP, card_b.K2_FLOOR, card_b.MIN_LIVE_START), ((128, 300), (100, 127), 128))
        self.assertEqual(card_b.SECTIONS, card_b.DEFAULT_SECTIONS + ('K2', 'X7', 'Z'))
        self.assertEqual(card_b.DEFAULT_SECTIONS, ('N', 'X', 'M', 'K', 'L', 'T'))
        self.assertTrue(set(card_b.CB2A_SECTIONS) <= set(card_b.DECISIVE_SECTIONS))
        for kind in ('k2_native_vs_extent', 'x7_narrow_vs_wide', 'x7_extent_vs_wide', 'z_trace_vs_eager',
                     'z_trace_vs_reference'):
            self.assertIn(kind, card_b.DECISIVE_KINDS)
        self.assertNotIn('k2_floor', card_b.DECISIVE_KINDS)
        self.assertEqual(sorted(card_b.NOT_RUN), ['K4'])
        self.assertEqual(card_b.RUN_ORDER[-4:], ('K2', 'X7', 'Z', 'timing'))

    def test_k2s_own_verdict(self):
        def entry(kind, differing=0, decisive=True):
            return probe.comparison(kind.split('_')[0].upper(), kind, 'x', differing, decisive)

        good = dict(sections=['K2', 'X7'], comparisons=[entry('k2_native_vs_extent'), entry('x7_narrow_vs_wide')],
                    liveness=[dict(section='K2', label='l', live=True)], failures=[])
        self.assertEqual(card_b.k2_verdict(good), 'PASS')
        self.assertEqual(card_b.k2_verdict(dict(good, sections=['X7'])), 'not_run')
        self.assertEqual(card_b.k2_verdict(dict(good, comparisons=good['comparisons'] + [
            entry('k2_native_vs_extent', 3)])), 'FAIL')
        self.assertEqual(card_b.k2_verdict(dict(good, comparisons=good['comparisons'] + [
            entry('k2_floor', 3, False)])), 'PASS')                                     # the floor is recorded
        other = dict(good, failures=['X7/seed0: RuntimeError: boom', 'Z/seed1/replay3/E256+0: not finite'])
        self.assertEqual(card_b.k2_verdict(other), 'PASS')                               # another section's failure
        self.assertEqual(card_b.decide(other)['verdict'], 'NO-DECISION')
        for failures in (['K2/seed0: RuntimeError: boom'], ['pool/seed1: boom'], ['the loaded _ttnncpp.so is x'],
                         ['factory log: no [QWEN-SDPA] line for flags=0x27 B=2 St=4104 mask_width_t=8']):
            self.assertEqual(card_b.k2_verdict(dict(good, failures=failures)), 'NO-DECISION', failures)
        self.assertEqual(card_b.k2_verdict(dict(good, error='terminated')), 'NO-DECISION')
        self.assertEqual(card_b.k2_verdict(dict(good, liveness=[dict(section='K2', label='l', live=False)])),
                         'NO-DECISION')
        self.assertEqual(card_b.k2_verdict(dict(good, liveness=[dict(section='X7', label='l', live=False)])), 'PASS')
        self.assertEqual(card_b.k2_verdict(dict(good, deadline=dict(skipped=['K2/seed3', 'X7/seed3']))),
                         'NO-DECISION')
        self.assertEqual(card_b.k2_verdict(dict(good, deadline=dict(skipped=['Z/seed4']))), 'PASS')
        self.assertEqual(card_b.k2_verdict(dict(good, comparisons=[entry('x7_narrow_vs_wide')])), 'NO-DECISION')
        good['decision'] = card_b.decide(good)
        good['k2_rows'] = dict(compared=32, equal=32, floor=8, floor_differing=5, capped=0)
        self.assertIn(' k2=1/1 k2_rows=32/32 k2_floor_differing=5/8 k2_verdict=PASS x7=1/1 z=none k4=not_run',
                      card_b.verdict_line(good))

    def test_the_cb2a_arguments(self):
        args = card_b.parse_args(['--out', 'x.json', '--sections', 'K2,X7,Z'])
        self.assertEqual((args.k2_sweep, args.k2_floor, args.cb2_extents, args.cb2_starts, args.z_families,
                          args.z_starts, args.output_memory),
                         ((128, 300), (100, 127), list(card_b.CB2_EXTENTS), list(card_b.CB2_STARTS),
                          list(card_b.Z_FAMILIES), list(card_b.Z_STARTS), 'l1'))
        self.assertEqual([card_b.run_tag(seed, name) for seed, name in card_b.section_runs(
            card_b.parse_args(['--out', 'x.json', '--sections', 'Z,K2,X7', '--seeds', '0,1']))],
            ['K2/seed0', 'X7/seed0', 'Z/seed0', 'timing/seed0', 'K2/seed1', 'X7/seed1', 'Z/seed1'])
        self.assertIsNone(card_b.parse_args(['--out', 'x.json', '--k2-floor', '']).k2_floor)
        # A K1 run on a small table never reads CB2a's served-geometry defaults.
        card_b.parse_args(['--out', 'x.json', '--capacity', '4352', '--extents', '2304'])
        for bad in (['--k2-sweep', '127:300'], ['--k2-sweep', '300'], ['--k2-sweep', '300:200'],
                    ['--k2-floor', '100:128'], ['--cb2-extents', '2300'], ['--cb2-extents', '2304,2304'],
                    ['--cb2-starts', '256'], ['--z-starts', ''], ['--z-families', '4000'],
                    ['--output-memory', 'sram'], ['--sections', 'K3'],
                    ['--capacity', '4352', '--extents', '2304', '--sections', 'Z', '--z-families', '4608'],
                    ['--capacity', '4352', '--extents', '2304', '--sections', 'X7'],
                    ['--capacity', '4352', '--extents', '2304', '--sections', 'K2', '--cb2-extents', '2304',
                     '--k2-sweep', '128:4352']):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                card_b.parse_args(['--out', 'x.json'] + bad)


SERVED_KWARGS = {'page_table_tensor', 'is_causal', 'attn_mask', 'scale', 'program_config', 'memory_config',
                 'cur_pos_tensor'}


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------

@unittest.skipUnless(base.BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.graft = base.make_graft(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def harness_args(self, **env):
        environ = {key: value for key, value in os.environ.items() if key not in base.SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='1',
                       KOPGRAFT64=self.graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(base.BINARY))
        environ.update(env)
        result = subprocess.run([base.BASH, base.RUNNER.as_posix()], env=environ, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        argv = shlex.split(line[0][len('### argv: '):])
        return card_b.parse_args(argv[argv.index('card') + 1:])

    def test_the_cb2a_passes(self):
        watcher = self.harness_args(WATCHER='1', CARD_B_ARGS='--sections K2,X7,Z')
        self.assertEqual((watcher.sections, watcher.seeds, watcher.variants, watcher.k2_sweep, watcher.k2_floor,
                          watcher.cb2_extents, watcher.cb2_starts, watcher.z_families, watcher.z_starts,
                          watcher.no_timing, watcher.watchdog, watcher.deadline_s, watcher.output_memory),
                         (['K2', 'X7', 'Z'], [0], ['normal'], (232, 263), (120, 127), [2304, 131328], [0, 240, 255],
                          [256, 512, 2304, 3840], list(card_b.Z_STARTS), True, 120.0, 2100.0, 'l1'))
        full = self.harness_args(CARD_B_ARGS='--sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky '
                                             '--no-timing')
        self.assertEqual((full.sections, full.seeds, full.variants, full.k2_sweep, full.k2_floor, full.cb2_extents,
                          full.cb2_starts, full.z_families, full.z_starts, full.no_timing, full.watchdog,
                          full.deadline_s),
                         (['K2', 'X7', 'Z'], [0, 1, 2, 3, 4], ['normal', 'peaky'], card_b.K2_SWEEP, card_b.K2_FLOOR,
                          list(card_b.CB2_EXTENTS), list(card_b.CB2_STARTS), list(card_b.Z_FAMILIES),
                          list(card_b.Z_STARTS), True, 300.0, 4800.0))
        self.assertEqual([card_b.run_tag(seed, name) for seed, name in card_b.section_runs(full)],
                         ['%s/seed%d' % (name, seed) for seed in range(5) for name in ('K2', 'X7', 'Z')])
        self.assertEqual(full.expect_binary_sha256, sha(base.BINARY))


# ---------------------------------------------------------------------------------------------
# The flow on the fake ttnn.
# ---------------------------------------------------------------------------------------------

SMALL = ['--k2-sweep', '232:263', '--k2-floor', '120:127', '--cb2-extents', '2304', '--cb2-starts', '7,255',
         '--z-families', '256,512,3840', '--z-starts', '0,32,255']


class DryRunTests(unittest.TestCase):
    """CB2a end to end on the fake: a 4,352-key table (17 chunks) standing for the served 131,328, K2's sweep and
    floor at the design's values, families 2,304 and 4,352 (= the table: E = C) for K2 and X7, Z at all 15 stale-writer
    families. One FULL run; each broken variant runs only the sections that must catch it, on SMALL (the CPU job's
    20-minute budget)."""

    BASE = ['--capacity', '4352', '--extents', '2304', '--seeds', '0', '--no-timing', '--sections', 'K2,X7,Z',
            '--cb2-extents', '2304,4352']

    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        graft = base.make_graft(self.dir)
        self.kernels = graft / 'sdpa_decode' / 'device' / 'kernels'
        self.binary = graft / '_ttnncpp.so'

    def tearDown(self):
        self.tmp.cleanup()

    def run_card(self, fake, extra=(), name='card', watchdog=None):
        out = self.dir / ('%s.json' % name)
        fake.report_path = out
        markers = dict(flags=True, share=True, stage1=False)
        argv = ['--out', str(out), '--kernel-root', str(self.kernels), '--expect-binary-sha256',
                sha(self.binary.read_bytes())]
        patches = [mock.patch.dict(sys.modules, {'ttnn': fake}),
                   mock.patch.object(card, 'loaded_binary', return_value=(str(self.binary), markers)),
                   mock.patch.dict(os.environ, {card.SCRATCH_ENV: '1'}), mock.patch.object(probe, 'clock', fake.clock),
                   mock.patch.object(card, 'WATCHDOG', card.WATCHDOG), mock.patch.object(probe, 'WATCHDOG', probe.WATCHDOG),
                   mock.patch.object(probe.k1, 'WATCHDOG', probe.k1.WATCHDOG),
                   mock.patch.object(probe, 'DEADLINE', probe.DEADLINE), mock.patch.object(probe, 'print', create=True),
                   mock.patch.object(card_b, 'print', create=True), mock.patch('sys.stdout')]
        if watchdog is not None:
            patches.append(mock.patch.object(probe.k1, 'Watchdog', watchdog))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            status = card_b.main(argv + self.BASE + list(extra))
        return status, json.loads(out.read_text())

    def kinds(self, report):
        return {kind: (row['equal'], row['runs']) for kind, row in report['tally'].items()}

    def test_cb2a_passes_end_to_end(self):
        torch = self.torch
        fake = FakeExtentTtnn(torch, record_calls=True)
        # peaky only (normal runs on SMALL in the variants below); Z at all 15 families, 3 of its 6 starts.
        status, report = self.run_card(fake, ['--variants', 'peaky', '--z-starts', '0,32,240'])
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict'], report['decision']['k2']),
                         (0, True, 'PASS', 'PASS'))
        self.assertEqual(report['sections_done'], ['K2/seed0', 'X7/seed0', 'Z/seed0'])
        self.assertEqual(report['cb2a']['k2_tickets'], dict(sweep=173, family=10, floor=28))
        kinds = self.kinds(report)
        self.assertEqual(kinds['k2_native_vs_extent'], (183, 183))                   # 173 sweep + 10 family
        self.assertEqual(kinds['k2_floor'][1], 28)
        # Rows: the sweep's 2,648 (15 short tickets at the cap) and 65 per family.
        self.assertEqual(report['k2_rows'], dict(compared=2778, equal=2778, floor=328,
                                                 floor_differing=report['k2_rows']['floor_differing'], capped=150))
        self.assertEqual((kinds['x7_narrow_vs_wide'], kinds['x7_extent_vs_wide']), ((10, 10), (10, 10)))
        self.assertEqual((kinds['z_trace_vs_eager'], kinds['z_trace_vs_reference']), ((45, 45), (45, 45)))
        self.assertEqual(report['z_families'], list(card_b.Z_FAMILIES))
        live = [entry for entry in report['liveness']]
        self.assertEqual(sum(entry['section'] == 'K2' for entry in live), 182)       # peaky, p + 1 below the table
        self.assertEqual(sum(entry['section'] == 'X7' for entry in live), 4)         # 2 families x 2 entries
        self.assertTrue(all(entry['live'] for entry in live))
        self.assertEqual(report['native_decode'], json.loads(json.dumps(card_b.NATIVE_DECODE)))
        self.assertEqual(report['unverified'], list(card_b.UNVERIFIED))
        self.assertTrue(report['extent_lines'])                                      # F22 after every 0x20 F4
        self.assertTrue(report['verdict_line'].startswith(
            'K64J_CARD verdict=PASS extent=none mixed=none share_slot0=none trace=none fence=none skip=none '
            'refusals=none live=186/186 families=0 k2=183/183 k2_rows=2778/2778 k2_floor_differing='),
            report['verdict_line'])
        self.assertIn(' k2_verdict=PASS x7=20/20 z=90/90 z_families=15 k4=not_run', report['verdict_line'])
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))
        # The calls: the native one exactly as tp.py passes it, the subject as the replay passes it plus cur_pos.
        native = [call for call in fake.recorded if call['program_config']['q_chunk_size'] == card.LEGACY]
        subject = [call for call in fake.recorded
                   if call['program_config']['q_chunk_size'] == card.MAGIC | card_b.SERVED_FLAGS]
        compile_time = [call for call in fake.recorded
                        if call['program_config']['q_chunk_size'] == card.MAGIC | card_b.COMPILE_FLAGS]
        self.assertEqual(len(fake.recorded), len(native) + len(subject) + len(compile_time))
        self.assertTrue(native and subject and compile_time)
        grid = dict(compute_with_storage_grid_size=(base.probe_tests.Grid.x, base.probe_tests.Grid.y))
        for call in native:
            self.assertEqual((call['positional'], call['options'], call['is_causal'], call['memory_config'],
                              call['program_config'], call['rows'], call['batches']),
                             (0, sorted(card_b.NATIVE_CALL_KWARGS), None, 'l1',
                              dict(grid, **card_b.NATIVE_PROGRAM_CONFIG), 12, 1))
        for group, expected in ((subject, SERVED_KWARGS), (compile_time, SERVED_KWARGS - {'cur_pos_tensor'})):
            for call in group:
                self.assertEqual((call['positional'], call['options'], call['is_causal'], call['memory_config'],
                                  call['rows'], call['batches']), (0, sorted(expected), False, 'l1', 96, 2))
                self.assertEqual(call['program_config'], dict(grid, exp_approx_mode=False, k_chunk_size=256,
                                                              q_chunk_size=call['program_config']['q_chunk_size']))

    def test_a_k2_reference_at_the_wrong_cur_pos_fails(self):
        for name, wrong in (('p+1', lambda position: position + 1), ('p-1', lambda position: position - 1),
                            ('E-1', lambda position: model.extent(position) - 1)):
            with self.subTest(wrong=name), mock.patch.object(card_b, 'native_cur_pos', wrong):
                status, report = self.run_card(FakeExtentTtnn(self.torch), ['--sections', 'K2'] + SMALL,
                                               name='wrong' + name)
                self.assertEqual((status, report['failures'], report['decision']['verdict'],
                                  report['decision']['k2']), (1, [], 'FAIL', 'FAIL'))
                tickets = [entry for entry in report['comparisons'] if entry['kind'] == 'k2_native_vs_extent']
                missed = [entry['label'] for entry in tickets if not entry['differing']]
                # E - 1 is right for the row at E - 1 only: the tickets whose one committed row is E - 1.
                expected = [entry['label'] for entry in tickets if entry['rows'] == 1] if name == 'E-1' else []
                self.assertEqual(missed, expected)
                self.assertIn(' k2_verdict=FAIL ', report['verdict_line'])

    def test_a_half_tile_q_that_moves_fails_k2(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'half_tile'}), ['--sections', 'K2,X7']
                                       + SMALL)
        self.assertEqual((report['failures'], report['decision']['verdict'], report['decision']['k2']),
                         ([], 'FAIL', 'FAIL'))
        kinds = self.kinds(report)
        self.assertEqual(kinds['k2_native_vs_extent'][0], 0)
        self.assertEqual(kinds['x7_narrow_vs_wide'], (2, 2))                          # X7 has no causal call

    def test_a_mask_read_at_the_capacity_fails_x7(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'tail_at_capacity'}),
                                       ['--sections', 'X7', '--cb2-extents', '2304,4352', '--cb2-starts', '0,240'])
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        kinds = self.kinds(report)
        self.assertEqual((kinds['x7_narrow_vs_wide'], kinds['x7_extent_vs_wide']), ((0, 4), (0, 4)))

    def test_a_narrow_mask_batch_offset_passes_k1_and_fails_x7_and_k2(self):
        broken = {'narrow_batch_offset'}
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken=broken),
                                       ['--sections', 'X', '--combos', 'G8B2:0x27', '--extents', '2304', '--starts',
                                        '7', '--variants', 'normal'], name='k1')
        self.assertEqual((report['decision']['verdict'], self.kinds(report)['extent_vs_reference']), ('PASS', (2, 2)))
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken=broken), ['--sections', 'X7'] + SMALL,
                                       name='x7')
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        # +7: the entries' tails differ, so entry 1 reading entry 0's rows moves it; +255: both tails mask nothing.
        self.assertEqual([entry['label'] for entry in report['comparisons'] if entry['differing']],
                         ['X7/seed0/E2304+7/normal/narrow', 'X7/seed0/E2304+7/normal/0x27'])
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken=broken), ['--sections', 'K2'] + SMALL,
                                       name='k2')
        self.assertEqual((report['decision']['verdict'], report['decision']['k2']), ('FAIL', 'FAIL'))

    def test_a_stale_writer_hangs_z_and_the_watchdog_fires(self):
        fired = {}
        out = self.dir / 'hang.json'

        class Firing(probe.k1.Watchdog):
            """The harness watchdog without its poll thread or faulthandler backstop, whose exit raises (after
            reading what on_fire wrote) instead of os._exit."""

            def __init__(self, seconds, on_fire=None):
                super().__init__(seconds, on_fire=on_fire, backstop=False, exit=self.stop)

            def start(self):
                return self

            def check(self):
                saved, sys.stdout = sys.stdout, io.StringIO()     # NativeLog's stdout is the real one: keep it quiet
                try:
                    return super().check()
                finally:
                    fired['printed'] = sys.stdout.getvalue()
                    sys.stdout = saved

            def stop(self, code):
                fired.update(code=code, report=json.loads(out.read_text()))
                raise SystemExit(code)

        fake = FakeExtentTtnn(self.torch, broken={'stale_writer'})
        with self.assertRaises(SystemExit) as caught:
            self.run_card(fake, ['--sections', 'Z', '--watchdog', '30', '--z-families', '256,512,3840', '--z-starts',
                                 '0,32'], name='hang', watchdog=Firing)
        self.assertEqual((caught.exception.code, fired['code'], fired['report']['passed']), (3, 3, False))
        self.assertEqual(fired['report']['error'], "watchdog: 'Z/seed0 warm eager' exceeded its budget")
        self.assertTrue(fired['printed'].startswith("WATCHDOG: 'Z/seed0 warm eager' did not return"), fired['printed'])
        self.assertTrue(fake.closed)
        # Above the zone the stale writer's tree is the live one: no hang, and the families agree.
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'stale_writer'}),
                                       ['--sections', 'X7', '--cb2-extents', '4352', '--cb2-starts', '7',
                                        '--watchdog', '30'], name='above', watchdog=Firing)
        self.assertEqual((status, report['decision']['verdict']), (0, 'PASS'))

    def test_a_section_that_raises_costs_only_its_own_evidence(self):
        with mock.patch.object(card_b, 'section_x7', side_effect=RuntimeError('boom')):
            status, report = self.run_card(FakeExtentTtnn(self.torch), ['--sections', 'K2,X7,Z'] + SMALL)
        self.assertEqual((report['sections_done'], report['sections_failed']), (['K2/seed0', 'Z/seed0'], ['X7/seed0']))
        self.assertEqual(report['failures'], ['X7/seed0: RuntimeError: boom'])
        self.assertEqual((report['decision']['verdict'], report['decision']['k2']), ('NO-DECISION', 'PASS'))
        # A program that never built (the fake's fatal_g8: every 96-row call FakeTtnn runs raises before its factory
        # line) leaves a requested program unlogged: a global failure, so K2 cannot stand either.
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'fatal_g8'}), ['--sections', 'K2,X7']
                                       + SMALL, name='fatal')
        self.assertEqual((report['sections_done'], report['sections_failed']), (['K2/seed0'], ['X7/seed0']))
        self.assertEqual((report['decision']['verdict'], report['decision']['k2']), ('NO-DECISION', 'NO-DECISION'))
        self.assertTrue(any('graft mounted, not executed' in failure for failure in report['failures']))

    def test_the_deadline_stops_cleanly_and_lists_the_rest(self):
        fake = FakeExtentTtnn(self.torch, seconds_per_call=1.0)
        status, report = self.run_card(fake, ['--sections', 'K2,X7,Z', '--deadline-s', '5'] + SMALL)
        self.assertEqual((status, report['decision']['verdict'], report['decision']['k2']),
                         (1, 'NO-DECISION', 'NO-DECISION'))
        self.assertEqual(report['deadline']['skipped'], ['K2/seed0', 'X7/seed0', 'Z/seed0'])
        self.assertTrue(report['deadline']['reached_at'].startswith('K2/seed0/normal/'), report['deadline'])
        self.assertTrue(fake.closed)
        fake = FakeExtentTtnn(self.torch, seconds_per_call=1.0)
        status, report = self.run_card(fake, ['--sections', 'Z', '--deadline-s', '6'] + SMALL, name='mid')
        self.assertEqual(report['deadline']['skipped'], ['Z/seed0'])
        self.assertTrue(report['deadline']['reached_at'].startswith('Z/seed0/replay'), report['deadline'])
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))      # the trace was released

    def test_sigterm_writes_the_partial_report_then_unwinds(self):
        fake = FakeExtentTtnn(self.torch, broken={'sigterm'})       # at the first non-causal FakeTtnn call: X7's
        status, report = self.run_card(fake, ['--sections', 'K2,X7'] + SMALL)
        self.assertEqual(status, 128 + int(signal.SIGTERM))
        self.assertEqual(fake.before_term['in_progress'], 'after K2/seed0')
        self.assertEqual((fake.after_term['in_progress'], fake.after_term['decision']['verdict']),
                         ('terminated', 'NO-DECISION'))
        self.assertTrue(fake.after_term['verdict_line'].startswith('K64J_CARD verdict=NO-DECISION'))
        self.assertEqual((report['sections_done'], report['decision']['k2']), (['K2/seed0'], 'NO-DECISION'))
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))


if __name__ == '__main__':
    unittest.main()
