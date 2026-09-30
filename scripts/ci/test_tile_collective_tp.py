"""tile_collective_tp on the CPU: the block's all-reduce split per 32-row tile at four cards, untouched everywhere else.

The fake all-reduce reproduces the property the four-card ring reduce-scatter has (reduce_scatter_common::
chunk_ring_parity): the direction, hence the association order, of every chunk of a tile is fixed by the tile's flat
place in the call's slice. It stamps each output row with the parities of its five 8-tile chunks (a 40-tile row of the
four-chip slice of 5,120 columns), so "the same row, reduced in the same order" is an equality of stamps."""

import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tile_collective_tp as collective
import tp_addresses
from tp_test_support import four_cards, pair

CHUNKS_PER_ROW_TILE = 5   # 40 tiles per row of the per-chip slice / an 8-tile chunk


class Tensor:
    def __init__(self, rows, width=5120, freed=None):
        self.rows = list(rows)
        self.width = width
        self.freed = False

    @property
    def shape(self):
        return (1, 1, len(self.rows), self.width)

    def memory_config(self):
        return 'DRAM'


class Operations:
    def __init__(self):
        self.calls = []

    def slice(self, tensor, start, stop, **kwargs):
        assert not tensor.freed
        assert start[0] == start[1] == 0 and start[3] == 0 and stop[3] == tensor.width
        self.calls.append(('slice', start[2], stop[2], kwargs))
        return Tensor(tensor.rows[start[2]:stop[2]], tensor.width)

    def concat(self, tensors, dim, **kwargs):
        assert dim == 2 and all(not tensor.freed for tensor in tensors)
        self.calls.append(('concat', len(tensors), kwargs))
        return Tensor([row for tensor in tensors for row in tensor.rows], tensors[0].width)

    def deallocate(self, tensor):
        self.calls.append(('deallocate', len(tensor.rows)))
        tensor.freed = True


class Ring:
    """The model's tt_all_reduce as far as the order of its sums goes; consumes its input like ccl.py."""

    def __init__(self):
        self.calls = []

    def __call__(self, tensor, mesh, ccl, cluster_axis=0, dim=3, topology='Ring', memory_config='DRAM'):
        assert not tensor.freed
        self.calls.append((len(tensor.rows), mesh, ccl, cluster_axis, dim, topology, memory_config))
        rows = []
        for index, label in enumerate(tensor.rows):
            tile = index // 32
            stamp = tuple(((tile * CHUNKS_PER_ROW_TILE + chunk) % 2) for chunk in range(CHUNKS_PER_ROW_TILE))
            rows.append((label, stamp))
        tensor.freed = True
        return Tensor(rows, tensor.width)


def block(rows):
    return Tensor(['r%d' % index for index in range(rows)])


def stamps(tensor):
    return [row[1] for row in tensor.rows]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.ring = Ring()
        self.operations = Operations()
        self.wrapper = collective.TileSplitAllReduce(self.ring, self.operations)
        self.arguments = ('mesh', 'ccl')
        self.options = dict(cluster_axis=0, dim=3, topology='Ring', memory_config='DRAM')

    def call(self, tensor):
        return self.wrapper(tensor, *self.arguments, **self.options)


class TheOrderTheRingGivesTheBlock(Fixture):
    """The failure this module exists for, in the fake: the unsplit block's second tile is reduced in another order."""

    def test_the_unsplit_second_tile_has_the_opposite_direction_of_the_one_tile_call(self):
        one_tile = self.call(block(32))
        whole = self.call(block(64))
        self.assertEqual(stamps(whole)[:32], stamps(one_tile))
        self.assertNotEqual(stamps(whole)[32:], stamps(one_tile))

    def test_the_sequential_engines_four_rows_are_the_first_tile(self):
        four = self.call(block(4))
        self.assertEqual(stamps(four), stamps(self.call(block(32)))[:4])

    def test_the_scope_makes_every_tile_the_one_tile_call(self):
        reference = stamps(self.call(block(32)))
        with collective.block_scope(64):
            joined = self.call(block(64))
        self.assertEqual(stamps(joined), reference + reference)
        self.assertEqual([row[0] for row in joined.rows], ['r%d' % index for index in range(64)])

    def test_four_tiles_are_four_one_tile_calls(self):
        reference = stamps(self.call(block(32)))
        with collective.block_scope(128):
            joined = self.call(block(128))
        self.assertEqual(stamps(joined), reference * 4)
        self.assertEqual([call[0] for call in self.ring.calls[1:]], [32, 32, 32, 32])


class TheScope(Fixture):
    def test_outside_a_scope_the_call_is_the_models_own_call(self):
        wide = block(64)
        result = self.call(wide)
        self.assertEqual(self.ring.calls, [(64, 'mesh', 'ccl', 0, 3, 'Ring', 'DRAM')])
        self.assertEqual(self.operations.calls, [])
        self.assertEqual(len(result.rows), 64)

    def test_prefill_and_other_shapes_inside_a_scope_pass_through(self):
        with collective.block_scope(64):
            self.call(block(2048))      # a prefill chunk
            self.call(block(32))        # the one-tile engines
            self.call(block(4))
            self.call(block(96))
        self.assertEqual([call[0] for call in self.ring.calls], [2048, 32, 4, 96])
        self.assertEqual(self.operations.calls, [])

    def test_a_split_passes_the_arguments_to_every_tile_call_unchanged(self):
        with collective.block_scope(64):
            self.call(block(64))
        self.assertEqual(self.ring.calls, [(32, 'mesh', 'ccl', 0, 3, 'Ring', 'DRAM')] * 2)

    def test_the_wrapper_consumes_its_input_and_frees_every_intermediate(self):
        wide = block(64)
        with collective.block_scope(64):
            joined = self.call(wide)
        self.assertTrue(wide.freed)
        self.assertFalse(joined.freed)
        kinds = [call[0] for call in self.operations.calls]
        self.assertEqual(kinds.count('slice'), 2)
        self.assertEqual(kinds.count('concat'), 1)
        # the two reduced tiles are freed after the join; the joined block is the caller's
        self.assertEqual([call for call in self.operations.calls if call[0] == 'deallocate'],
                         [('deallocate', 64), ('deallocate', 32), ('deallocate', 32)])
        self.assertEqual(self.operations.calls[0][3], dict(memory_config='DRAM'))
        self.assertEqual([call[2] for call in self.operations.calls if call[0] == 'concat'], [dict(memory_config='DRAM')])

    def test_the_slices_are_whole_tiles_in_row_order(self):
        with collective.block_scope(96):
            self.call(block(96))
        self.assertEqual([call[1:3] for call in self.operations.calls if call[0] == 'slice'],
                         [(0, 32), (32, 64), (64, 96)])

    def test_a_failing_tile_frees_the_tiles_after_it_and_the_ones_already_reduced(self):
        wide = block(96)
        pieces = []
        real = self.operations.slice

        def spy(*args, **kwargs):
            pieces.append(real(*args, **kwargs))
            return pieces[-1]
        self.operations.slice = spy
        good = self.ring

        def fail_second(tensor, *args, **kwargs):
            if len(good.calls) == 1:
                raise RuntimeError('collective died')
            return good(tensor, *args, **kwargs)
        self.wrapper = collective.TileSplitAllReduce(fail_second, self.operations)
        with collective.block_scope(96):
            with self.assertRaises(RuntimeError):
                self.call(wide)
        self.assertTrue(pieces[2].freed)                 # the tile after the one that failed

    def test_a_failing_slice_leaves_the_input_alone_and_frees_the_cut_tiles(self):
        wide = block(64)
        cut = []
        real = self.operations.slice

        def spy(tensor, start, stop, **kwargs):
            if start[2] == 32:
                raise RuntimeError('no memory')
            cut.append(real(tensor, start, stop, **kwargs))
            return cut[-1]
        self.operations.slice = spy
        with collective.block_scope(64):
            with self.assertRaises(RuntimeError):
                self.call(wide)
        self.assertFalse(wide.freed)
        self.assertTrue(cut[0].freed)

    def test_only_whole_tiles_beyond_one_are_a_block(self):
        for rows in (32, 48, 33, 0, -32, 64.0, True):
            with self.assertRaises(ValueError, msg=rows):
                with collective.block_scope(rows):
                    pass

    def test_scopes_do_not_nest_and_close_on_error(self):
        with collective.block_scope(64):
            with self.assertRaises(ValueError):
                with collective.block_scope(64):
                    pass
        self.call(block(64))                                  # closed again: this one passes through
        self.assertEqual(self.ring.calls[-1][0], 64)
        with self.assertRaises(RuntimeError):
            with collective.block_scope(64):
                raise RuntimeError('forward failed')
        self.call(block(64))
        self.assertEqual(self.ring.calls[-1][0], 64)

    def test_the_guard_refuses_a_round_that_split_fewer_than_expected(self):
        with self.assertRaises(AssertionError) as raised:
            with collective.block_scope(64, expected=2):
                self.call(block(64))
        self.assertIn('1 times', str(raised.exception).replace('engaged ', ''))

    def test_the_guard_refuses_a_round_that_never_reached_the_wrapper(self):
        with self.assertRaises(AssertionError):
            with collective.block_scope(64, expected=128):
                pass

    def test_the_guard_passes_when_all_are_split_and_logs_it(self):
        lines = []
        with collective.block_scope(64, expected=3, log=lambda text, *values: lines.append(text.format(*values))):
            for _ in range(3):
                self.call(block(64))
        self.assertEqual(lines, ['[PINDIAG] tile-split all-reduce: 3 of 64 rows in 2 tiles each'])

    def test_an_error_in_the_forward_is_not_replaced_by_the_guard(self):
        with self.assertRaises(RuntimeError):
            with collective.block_scope(64, expected=128):
                raise RuntimeError('forward failed')

    def test_a_wrapper_is_never_wrapped_twice(self):
        with self.assertRaises(ValueError):
            collective.TileSplitAllReduce(self.wrapper)
        with self.assertRaises(ValueError):
            collective.TileSplitAllReduce(None)


class TheInstall(unittest.TestCase):
    def setUp(self):
        self.original = Ring()
        self.ccl = types.ModuleType('fake_ccl')
        self.ccl.tt_all_reduce = self.original
        self.holder = types.ModuleType('fake_graft_attention')
        self.holder.tt_all_reduce = self.original
        self.other = types.ModuleType('fake_graft_other')
        self.other.tt_all_reduce = lambda *args: None          # a same-named function of another module
        self.alias = types.ModuleType('fake_graft_alias')
        self.alias.reduce = self.original                      # an alias under another name
        for module in (self.ccl, self.holder, self.other, self.alias):
            sys.modules[module.__name__] = module
        self.addCleanup(lambda: [sys.modules.pop(module.__name__, None)
                                 for module in (self.ccl, self.holder, self.other, self.alias)])
        import model_batch
        self.model_batch = model_batch
        self.run_before = model_batch.ModelBatch.run
        self.addCleanup(lambda: setattr(model_batch.ModelBatch, 'run', self.run_before))

    def test_the_ccl_name_and_every_module_holding_the_function_are_rebound(self):
        changed = collective.install(self.ccl, scope=False)
        self.assertIsInstance(self.ccl.tt_all_reduce, collective.TileSplitAllReduce)
        self.assertIs(self.holder.tt_all_reduce, self.ccl.tt_all_reduce)
        self.assertIs(self.ccl.tt_all_reduce.original, self.original)
        self.assertIsNot(self.other.tt_all_reduce, self.ccl.tt_all_reduce)
        self.assertIs(self.alias.reduce, self.original)
        self.assertEqual(sorted(namespace['__name__'] for namespace, _, _ in changed),
                         ['fake_ccl', 'fake_graft_attention'])

    def test_a_second_install_changes_nothing(self):
        collective.install(self.ccl)
        self.assertEqual(collective.install(self.ccl), [])
        self.assertEqual(collective.install_scope(), [])

    def test_without_the_model_tree_nothing_is_bound_and_model_batch_run_is_the_pinned_method(self):
        with patch.dict(sys.modules, {'models': None}):
            self.assertEqual(collective.install(), [])
        self.assertIs(self.model_batch.ModelBatch.run, self.run_before)

    def test_the_install_wraps_model_batch_run_and_uninstall_puts_the_pinned_method_back(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}), patch.object(collective, 'CCL_MODULE', 'fake_ccl'):
            tp_addresses.install()
            try:
                wrapped = self.model_batch.ModelBatch.run
                self.assertIsNot(wrapped, self.run_before)
                self.assertIs(wrapped.tile_scope_of, self.run_before)
                self.assertIsInstance(self.holder.tt_all_reduce, collective.TileSplitAllReduce)
            finally:
                tp_addresses.uninstall()
        self.assertIs(self.model_batch.ModelBatch.run, self.run_before)
        self.assertIs(self.holder.tt_all_reduce, self.original)
        self.assertIs(self.ccl.tt_all_reduce, self.original)

    def test_the_pair_never_binds_it(self):
        with pair(), patch.object(collective, 'CCL_MODULE', 'fake_ccl'):
            with self.assertRaises(ValueError):
                tp_addresses.install()
        self.assertIs(self.ccl.tt_all_reduce, self.original)
        self.assertIs(self.model_batch.ModelBatch.run, self.run_before)


class TheBlocksScope(unittest.TestCase):
    """scope_for and scoped_run: a ModelBatch forward inside the block scope, only for a block wider than a tile."""

    def batch(self, rows, native_m3=True, layers=64):
        return types.SimpleNamespace(rows=rows, native_m3=native_m3, model=types.SimpleNamespace(layers=[None] * layers))

    def test_within_one_tile_there_is_nothing_to_split(self):
        for rows in (1, 4, 16, 32):
            with collective.scope_for(self.batch(rows)):
                self.assertIsNone(collective._STATE['rows'])

    def test_a_wide_block_splits_and_expects_all_128_reductions(self):
        with self.assertRaises(AssertionError) as raised:
            with collective.scope_for(self.batch(64)):
                self.assertEqual(collective._STATE['rows'], 64)
        self.assertIn('128 expected', str(raised.exception))

    def test_without_native_m3_the_mlp_reduces_in_two_tile_calls_already(self):
        with self.assertRaises(AssertionError) as raised:
            with collective.scope_for(self.batch(64, native_m3=False)):
                pass
        self.assertIn('64 expected', str(raised.exception))

    def test_a_run_is_called_unchanged_inside_the_scope_and_returns_its_result(self):
        seen = []
        ring = Ring()
        wrapper = collective.TileSplitAllReduce(ring, Operations())

        def run(self, *args, **kwargs):
            seen.append((args, kwargs, collective._STATE['rows']))
            for _ in range(128):
                wrapper(block(64), 'mesh', 'ccl', cluster_axis=0, dim=3, topology='Ring', memory_config='DRAM')
            return 'result'

        scoped = collective.scoped_run(run)
        self.assertEqual(scoped(self.batch(64), sharded_logits=True), 'result')
        self.assertEqual(seen, [((), {'sharded_logits': True}, 64)])
        self.assertIsNone(collective._STATE['rows'])
        self.assertIs(scoped.tile_scope_of, run)

    def test_a_run_that_never_reaches_the_wrapper_is_refused(self):
        scoped = collective.scoped_run(lambda self, **kwargs: 'unsplit result')
        with self.assertRaises(AssertionError):
            scoped(self.batch(64))
        self.assertEqual(scoped(self.batch(32)), 'unsplit result')

    def test_the_pinned_run_signature_and_call_are_unchanged(self):
        import inspect
        import model_batch
        source = inspect.getsource(model_batch.ModelBatch.run)
        self.assertIn('with instance_overrides(self.bindings), mask_scope:', source)
        self.assertIn("**({'sharded_lm_head': True} if sharded_logits else {})", source)
        self.assertNotIn('collective', source)
        self.assertEqual(list(inspect.signature(model_batch.ModelBatch.run).parameters), ['self', 'sharded_logits'])


if __name__ == '__main__':
    unittest.main()
