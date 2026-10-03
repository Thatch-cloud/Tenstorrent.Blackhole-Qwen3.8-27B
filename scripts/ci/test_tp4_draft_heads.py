"""tp4_draft_heads (QWEN_FAST_TP4_DRAFT_HEADS, D2c): the drafter's head split and merge as tile permutations.

nlp_create_qkv_heads (transpose_k_heads=False) and nlp_concat_heads on tiled bf16 with head_dim 128 (four whole tiles) move no element
inside a tile. The tests apply the planner's tile moves to a matrix cut into 32 x 32 blocks and compare the result with the torch
reshape / permute reference of both ops at the drafter's per-chip head counts (8 query, 2 KV at four cards), for the pair path's
shapes (32-row query, 32 to 2,080 key rows) and the quad's (64 rows); plus the launch the builder describes, the refusals and the
twins' flag-off path."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import draft_head_layout_tp as pair_twin
import quad_draft_tp as quad_twin
import tp4_draft_conv as conv
import tp4_draft_heads as heads
import tp4_sampdraft
from test_tp4_shard_argmax import FakeOperations

HEADS, KV_HEADS, DIM = 8, 2, 128


def blocks(matrix):
    """{(row tile, column tile): 32 x 32 block} of a (rows, width) matrix."""
    rows, width = matrix.shape
    return {(rt, ct): matrix[32 * rt:32 * rt + 32, 32 * ct:32 * ct + 32].clone()
            for rt in range(rows // 32) for ct in range(width // 32)}


def source_page(slot_shape, rt, ct):
    return rt * (slot_shape[1] // 32) + ct


def assemble_heads(pages, count, rows):
    """A (count, rows, 128) tensor from {page: block} of a heads tensor (page = (h * row tiles + rt) * 4 + dt)."""
    row_tiles = rows // 32
    out = torch.zeros(count, rows, DIM)
    for head in range(count):
        for rt in range(row_tiles):
            for dt in range(4):
                out[head, 32 * rt:32 * rt + 32, 32 * dt:32 * dt + 32] = pages[(head * row_tiles + rt) * 4 + dt]
    return out


def run_split(query, key, value):
    """Apply the planner's moves to the three projections; return the three heads tensors."""
    sources = [query, key, value]
    pages = [{source_page(source.shape, rt, ct): block for (rt, ct), block in blocks(source).items()} for source in sources]
    moved = [dict(), dict(), dict()]
    for slot, page, destination_slot, destination_page in heads.split_tasks(HEADS, KV_HEADS, query.shape[0], key.shape[0]):
        assert slot == destination_slot
        assert destination_page not in moved[destination_slot]
        moved[destination_slot][destination_page] = pages[slot][page]
    return (assemble_heads(moved[0], HEADS, query.shape[0]), assemble_heads(moved[1], KV_HEADS, key.shape[0]),
            assemble_heads(moved[2], KV_HEADS, value.shape[0]))


def reference_split(query, key, value):
    """nlp_create_qkv_heads on query (rows, heads * 128) and the K | V projection: (heads, rows, 128) each."""
    q = query.reshape(query.shape[0], HEADS, DIM).permute(1, 0, 2)
    k = key.reshape(key.shape[0], KV_HEADS, DIM).permute(1, 0, 2)
    v = value.reshape(value.shape[0], KV_HEADS, DIM).permute(1, 0, 2)
    return q, k, v


class PermutationTests(unittest.TestCase):
    def draw(self, query_rows, kv_rows, seed=0):
        generator = torch.Generator().manual_seed(seed)
        return (torch.randn(query_rows, HEADS * DIM, generator=generator), torch.randn(kv_rows, KV_HEADS * DIM, generator=generator),
                torch.randn(kv_rows, KV_HEADS * DIM, generator=generator))

    def test_the_split_is_nlp_create_qkv_heads_for_the_pair_and_quad_shapes(self):
        for query_rows, kv_rows in ((32, 32), (32, 64), (32, 96), (32, 2080), (64, 64)):
            with self.subTest(query_rows=query_rows, kv_rows=kv_rows):
                query, key, value = self.draw(query_rows, kv_rows, seed=kv_rows)
                got = run_split(query, key, value)
                want = reference_split(query, key, value)
                for mine, theirs in zip(got, want):
                    self.assertTrue(torch.equal(mine, theirs))

    def test_the_merge_is_nlp_concat_heads_and_the_inverse_of_the_query_split(self):
        for rows in (32, 64):
            tensor = torch.randn(HEADS, rows, DIM)
            pages = {}
            for head in range(HEADS):
                for rt in range(rows // 32):
                    for dt in range(4):
                        pages[(head * (rows // 32) + rt) * 4 + dt] = tensor[head, 32 * rt:32 * rt + 32, 32 * dt:32 * dt + 32]
            out = {}
            for slot, page, destination_slot, destination_page in heads.merge_tasks(HEADS, rows):
                self.assertEqual((slot, destination_slot), (0, 0))
                out[destination_page] = pages[page]
            width_tiles = HEADS * 4
            merged = torch.zeros(rows, HEADS * DIM)
            for page, block in out.items():
                rt, ct = divmod(page, width_tiles)
                merged[32 * rt:32 * rt + 32, 32 * ct:32 * ct + 32] = block
            self.assertTrue(torch.equal(merged, tensor.permute(1, 0, 2).reshape(rows, HEADS * DIM)))     # nlp_concat_heads

    def test_every_destination_tile_is_written_exactly_once_and_every_source_tile_read_once(self):
        for query_rows, kv_rows in ((32, 32), (32, 2080), (64, 64)):
            tasks = heads.split_tasks(HEADS, KV_HEADS, query_rows, kv_rows)
            expected = {0: HEADS * (query_rows // 32) * 4, 1: KV_HEADS * (kv_rows // 32) * 4, 2: KV_HEADS * (kv_rows // 32) * 4}
            for slot, count in expected.items():
                destinations = sorted(task[3] for task in tasks if task[2] == slot)
                sources = sorted(task[1] for task in tasks if task[0] == slot)
                self.assertEqual(destinations, list(range(count)))
                self.assertEqual(sources, list(range(count)))

    def test_rows_that_are_not_whole_tiles_are_refused(self):
        for rows in (0, 16, 33, 2081, 32.0, None):
            with self.assertRaises(heads.Unsupported):
                heads.split_tasks(HEADS, KV_HEADS, 32, rows)
            with self.assertRaises(heads.Unsupported):
                heads.merge_tasks(HEADS, rows)


class DistributionTests(unittest.TestCase):
    def test_runs_are_contiguous_cover_every_task_once_and_fit_the_argument_budget(self):
        for query_rows, kv_rows in ((32, 32), (32, 2080), (64, 64)):
            tasks = heads.split_tasks(HEADS, KV_HEADS, query_rows, kv_rows)
            runs = heads.distribute(tasks, heads.MAX_CORES)
            self.assertLessEqual(len(runs), heads.MAX_CORES)
            self.assertEqual([task for run in runs for task in run], tasks)
            self.assertLessEqual(1 + heads.TASK_WORDS * heads.capacity(runs), heads.MAX_ARGUMENT_WORDS)

    def test_a_launch_that_does_not_fit_is_refused(self):
        with self.assertRaises(heads.Unsupported):
            heads.distribute([(0, index, 0, index) for index in range(24 * 80)], 24)
        with self.assertRaises(heads.Unsupported):
            heads.distribute([], 24)

    def test_every_core_is_padded_to_the_capacity_so_the_program_cache_key_carries_the_length(self):
        runs = heads.distribute(heads.split_tasks(HEADS, KV_HEADS, 32, 96), heads.MAX_CORES)
        arguments = heads.runtime_arguments(runs, [100, 200, 300], [1000, 2000, 3000])
        self.assertEqual({len(words) for words in arguments}, {1 + heads.TASK_WORDS * heads.capacity(runs)})
        self.assertEqual([words[0] for words in arguments], [len(run) for run in runs])
        first = runs[0][0]
        self.assertEqual(arguments[0][1:5], [[100, 200, 300][first[0]], first[1], [1000, 2000, 3000][first[2]], first[3]])


def tile(operations, shape):
    return operations.tensor(shape)


class Setup(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.DRAFT_HEADS: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        tp4_sampdraft._LOGGED.clear()
        self.lines = []
        logger = patch.object(tp4_sampdraft, 'log_line', side_effect=self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)
        self.owned = []

    def retain(self, tensor):
        self.owned.append(tensor)
        return tensor


class LaunchTests(Setup):
    def test_the_split_is_one_launch_and_retains_the_three_heads(self):
        operations = FakeOperations()
        query, key, value = (tile(operations, (1, 1, 32, 1024)), tile(operations, (1, 1, 96, 256)), tile(operations, (1, 1, 96, 256)))
        served = Mock()
        result = heads.split_heads(operations, query, key, value, self.retain, served=served, site='pair')
        served.assert_not_called()
        self.assertEqual([tensor.shape for tensor in operations.empties], [(1, 8, 32, 128), (1, 2, 96, 128), (1, 2, 96, 128)])
        self.assertEqual(self.owned, operations.empties)
        self.assertEqual((result['q'], result['k'], result['v']), tuple(operations.empties))
        self.assertEqual(len(operations.generic), 1)
        tensors, program = operations.generic[0]
        self.assertEqual(tensors, [query, key, value, *operations.empties])
        runs = heads.distribute(heads.split_tasks(8, 2, 32, 96), 24)
        for chip, descriptor in enumerate(program.values()):
            kernel, = descriptor['kernels']
            self.assertTrue(kernel['kernel_source'].endswith('draft_heads_copy.cpp'))
            self.assertEqual(kernel['compile_time_args'][-1], heads.capacity(runs))
            self.assertEqual(len(kernel['compile_time_args']), 3)
            self.assertEqual(descriptor['cbs'][0]['total_size'], 8 * 2048)
            cores = [(x, y) for x in kernel['runtime_args'] for y in kernel['runtime_args'][x]]
            self.assertEqual(len(cores), len(runs))
            sizes = {len(words) for column in kernel['runtime_args'].values() for words in column.values()}
            self.assertEqual(sizes, {1 + 4 * heads.capacity(runs)})
            first = kernel['runtime_args'][0][0]
            self.assertEqual(first[1], query.shards[chip].buffer_address())
            self.assertEqual(first[3], operations.empties[0].shards[chip].buffer_address())
        self.assertEqual([line for line in self.lines if line.startswith(tp4_sampdraft.HEADS_ENGAGED)],
                         ['%s site=pair op=split rows=32x96 heads=8/2 tiles=%d cores=%d' % (
                             tp4_sampdraft.HEADS_ENGAGED, 8 * 4 + 2 * 3 * 4 * 2, len(runs))])

    def test_the_merge_is_one_launch_and_retains_the_result(self):
        operations = FakeOperations()
        value = tile(operations, (1, 8, 64, 128))
        result = heads.merge_heads(operations, value, self.retain, served=Mock(), site='quad')
        self.assertEqual(result.shape, (1, 1, 64, 1024))
        self.assertEqual(self.owned, [result])
        tensors, program = operations.generic[0]
        self.assertEqual(tensors, [value, result])

    def test_a_call_it_cannot_take_runs_the_served_function_and_says_why_once(self):
        operations = FakeOperations()
        cases = [((1, 1, 32, 1024), (1, 1, 48, 256)),        # key rows are not whole tiles
                 ((1, 1, 32, 512), (1, 1, 32, 256)),         # a query of another width
                 ((32, 1024), (1, 1, 32, 256))]
        for query_shape, key_shape in cases:
            served = Mock(return_value='served')
            query, key, value = tile(operations, query_shape), tile(operations, key_shape), tile(operations, key_shape)
            self.assertEqual(heads.split_heads(operations, query, key, value, self.retain, served=served, site='pair'), 'served')
            served.assert_called_once()
        bad = tile(operations, (1, 1, 32, 1024))
        bad.dtype = 'bf8'
        served = Mock(return_value='served')
        self.assertEqual(heads.split_heads(operations, bad, tile(operations, (1, 1, 32, 256)), tile(operations, (1, 1, 32, 256)),
                                           self.retain, served=served, site='pair'), 'served')
        self.assertEqual(heads.merge_heads(operations, tile(operations, (1, 4, 32, 128)), self.retain, served=served, site='pair'),
                         'served')
        self.assertEqual(operations.generic, [])
        self.assertEqual(operations.empties, [])
        self.assertTrue(all(line.startswith(tp4_sampdraft.HEADS_FALLBACK) for line in self.lines))
        self.assertEqual(len(self.lines), 5)

    def test_a_failed_launch_frees_the_outputs(self):
        operations = FakeOperations()
        operations.generic_op = Mock(side_effect=RuntimeError('submit'))
        query, key, value = tile(operations, (1, 1, 32, 1024)), tile(operations, (1, 1, 32, 256)), tile(operations, (1, 1, 32, 256))
        with self.assertRaises(RuntimeError):
            heads.split_heads(operations, query, key, value, self.retain, served=Mock(), site='pair')
        self.assertEqual(operations.freed, operations.empties)
        self.assertEqual(self.owned, [])


class AuditTests(Setup):
    """QWEN_FAST_TP4_DRAFT_HEADS_AUDIT: inside a drafter capture scope the served split / merge runs beside the launch and each output pair
    is held (the engaged output cloned) for the bucket's compare; outside a scope nothing is audited."""

    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {tp4_sampdraft.DRAFT_HEADS_AUDIT: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        conv._CURRENT['scope'] = None
        conv._OPENED.clear()
        self.addCleanup(lambda: conv._CURRENT.update(scope=None))

    def operations(self):
        operations = FakeOperations()
        operations.clone = Mock(side_effect=lambda value, memory_config=None: operations.tensor(value.shape))
        return operations

    def split(self, operations):
        query, key, value = tile(operations, (1, 1, 32, 1024)), tile(operations, (1, 1, 32, 256)), tile(operations, (1, 1, 32, 256))
        references = [tile(operations, (1, 8, 32, 128)), tile(operations, (1, 2, 32, 128)), tile(operations, (1, 2, 32, 128))]
        served = Mock(return_value=dict(q=references[0], k=references[1], v=references[2]))
        return heads.split_heads(operations, query, key, value, self.retain, served=served, site='pair'), served, references

    def test_inside_a_scope_the_split_pairs_each_head_tensor_with_the_served_one(self):
        operations = self.operations()
        scope = conv.open_scope('pair')
        try:
            result, served, references = self.split(operations)
        finally:
            conv.close_scope(scope)
        served.assert_called_once()
        self.assertEqual(len(scope), 3)
        self.assertEqual([pair[0] for pair in scope.pairs], ['heads'] * 3)
        self.assertEqual([pair[2] for pair in scope.pairs], references)
        self.assertEqual([pair[3] for pair in scope.pairs], [False] * 3)          # the caller's retain owns the served outputs
        self.assertEqual(operations.clone.call_count, 3)
        self.assertEqual([tensor.shape for tensor in (result['q'], result['k'], result['v'])],
                         [(1, 8, 32, 128), (1, 2, 32, 128), (1, 2, 32, 128)])

    def test_inside_a_scope_the_merge_pairs_its_result(self):
        operations = self.operations()
        reference = tile(operations, (1, 1, 32, 1024))
        scope = conv.open_scope('quad')
        try:
            result = heads.merge_heads(operations, tile(operations, (1, 8, 32, 128)), self.retain,
                                       served=Mock(return_value=reference), site='quad')
        finally:
            conv.close_scope(scope)
        self.assertEqual(len(scope), 1)
        self.assertIs(scope.pairs[0][2], reference)
        self.assertIsNot(result, reference)

    def test_with_no_scope_the_served_function_is_not_run_beside_the_launch(self):
        operations = self.operations()
        result, served, _ = self.split(operations)
        served.assert_not_called()
        self.assertEqual(operations.clone.call_count, 0)

    def test_the_pairs_compare_and_release_without_freeing_the_callers_tensors(self):
        operations = self.operations()
        scope = conv.open_scope('pair')
        try:
            _, _, references = self.split(operations)
        finally:
            conv.close_scope(scope)
        operations.to_torch = lambda part: torch.zeros(1, 1, 32, 1024, dtype=torch.bfloat16)
        self.assertEqual(conv.compare_scope(operations, scope), 3)
        self.assertTrue(any(line.startswith('%s exact=True tensors=3 round=1' % tp4_sampdraft.HEADS_AUDIT) for line in self.lines))
        clones = [pair[1] for pair in scope.pairs]
        conv.release_scope(operations, scope)
        self.assertEqual(operations.freed, clones)
        for reference in references:
            self.assertNotIn(reference, operations.freed)

    def test_a_differing_head_tensor_logs_the_heads_mismatch_marker_and_raises(self):
        operations = self.operations()
        scope = conv.open_scope('pair')
        try:
            self.split(operations)
        finally:
            conv.close_scope(scope)
        _, mine, served, _ = scope.pairs[1]
        mine_ids = {id(part) for part in mine.shards}
        operations.to_torch = lambda part: torch.full((1, 1, 32, 128), 1.0 if id(part) in mine_ids else 2.0, dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            conv.compare_scope(operations, scope)
        self.assertTrue(self.lines[-1].startswith(tp4_sampdraft.HEADS_MISMATCH))

    def test_the_audit_flag_without_the_lever_is_refused(self):
        with patch.dict(os.environ, {tp4_sampdraft.DRAFT_HEADS: '0'}):
            with self.assertRaises(ValueError):
                tp4_sampdraft.audit_enabled(tp4_sampdraft.DRAFT_HEADS_AUDIT)


class TwinTests(unittest.TestCase):
    def environment(self, environ):
        patcher = patch.dict(os.environ, environ, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_flag_off_both_twins_call_their_served_functions(self):
        self.environment({'QWEN_FAST_TP': '4'})
        os.environ.pop(tp4_sampdraft.DRAFT_HEADS, None)
        for module in (pair_twin, quad_twin):
            with patch.object(module, 'served_split_projected_heads', return_value='split') as split, \
                    patch.object(module, 'served_concatenate_query_heads', return_value='merge') as merge, \
                    patch('tp4_draft_heads.split_heads') as fast_split, patch('tp4_draft_heads.merge_heads') as fast_merge:
                self.assertEqual(module.split_projected_heads('ops', 'q', 'k', 'v', 'retain'), 'split')
                self.assertEqual(module.concatenate_query_heads('ops', 'x', 'retain'), 'merge')
            split.assert_called_once_with('ops', 'q', 'k', 'v', 'retain')
            merge.assert_called_once_with('ops', 'x', 'retain')
            fast_split.assert_not_called()
            fast_merge.assert_not_called()

    def test_flag_on_both_twins_call_the_tile_copy_with_their_site_and_the_served_fallback(self):
        self.environment({'QWEN_FAST_TP': '4', tp4_sampdraft.DRAFT_HEADS: '1'})
        for module, site in ((pair_twin, 'pair'), (quad_twin, 'quad')):
            with patch.object(module, 'served_split_projected_heads', return_value='split') as split, \
                    patch.object(module, 'served_concatenate_query_heads', return_value='merge') as merge, \
                    patch('tp4_draft_heads.split_heads', return_value='fast split') as fast_split, \
                    patch('tp4_draft_heads.merge_heads', return_value='fast merge') as fast_merge:
                self.assertEqual(module.split_projected_heads('ops', 'q', 'k', 'v', 'retain'), 'fast split')
                self.assertEqual(module.concatenate_query_heads('ops', 'x', 'retain'), 'fast merge')
                self.assertEqual(fast_split.call_args.kwargs['site'], site)
                self.assertEqual(fast_merge.call_args.kwargs['site'], site)
                self.assertEqual(fast_split.call_args.kwargs['served'](), 'split')
                self.assertEqual(fast_merge.call_args.kwargs['served'](), 'merge')
            split.assert_called_once_with('ops', 'q', 'k', 'v', 'retain')
            merge.assert_called_once_with('ops', 'x', 'retain')

    def test_the_flag_is_refused_at_the_pair(self):
        self.environment({tp4_sampdraft.DRAFT_HEADS: '1'})
        os.environ.pop('QWEN_FAST_TP', None)
        with self.assertRaisesRegex(ValueError, 'TP4 levers'):
            pair_twin.split_projected_heads('ops', 'q', 'k', 'v', 'retain')


class KernelSourceTests(unittest.TestCase):
    def test_the_kernel_constants_agree_with_the_planner(self):
        from pathlib import Path
        text = Path(heads.__file__).with_name('draft_heads_copy.cpp').read_text()
        self.assertIn('constexpr uint32_t LANES = %d;' % heads.LANES, text)
        self.assertIn('constexpr uint32_t TASK_WORDS = %d;' % heads.TASK_WORDS, text)
        self.assertIn('constexpr uint32_t TILE_BYTES = %d;' % heads.TILE_BYTES, text)
        self.assertIn('given < CAPACITY ? given : CAPACITY', text)
        for argument in ('get_arg_val<uint32_t>(base)', 'get_arg_val<uint32_t>(base + 1)', 'get_arg_val<uint32_t>(base + 2)',
                         'get_arg_val<uint32_t>(base + 3)'):
            self.assertIn(argument, text)


if __name__ == '__main__':
    unittest.main()
