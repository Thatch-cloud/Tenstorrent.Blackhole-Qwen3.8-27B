from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from force_argmax import sample_rows


class ForceArgmaxTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(tt_sampling=SimpleNamespace(force_argmax_sampling=True, max_batch_size=32),
            seed_manager=SimpleNamespace(has_active_request_seed=Mock(return_value=False)),
            sample=Mock(return_value=('ids', None))), SimpleNamespace(pad=Mock(return_value='padded'), deallocate=Mock(),
                get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2)

    def test_all_widths_use_one_untraced_sampler_and_own_only_padding(self):
        for rows in (1, 2, 4, 8, 16, 32):
            sampler, operations = self.fixture()
            logits = SimpleNamespace(shape=(1, 1, rows, 124160))
            self.assertEqual(sample_rows(sampler, logits, rows, operations), 'ids')
            sampler.sample.assert_called_once_with(logits if rows == 32 else 'padded', enable_trace=False)
            if rows == 32:
                operations.pad.assert_not_called()
                operations.deallocate.assert_not_called()
            else:
                operations.pad.assert_called_once_with(logits, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], value=0.0)
                operations.deallocate.assert_called_once_with('padded')

    def test_rejects_sampling_mode_seed_or_geometry_before_device_ops(self):
        for invalid in ('mode', 'batch', 'seed', 'shape', 'width', 'width-48', 'width-128'):
            sampler, operations = self.fixture()
            logits = SimpleNamespace(shape=(1, 1, 16, 124160))
            if invalid == 'mode':
                sampler.tt_sampling.force_argmax_sampling = False
            elif invalid == 'batch':
                sampler.tt_sampling.max_batch_size = 64
            elif invalid == 'seed':
                sampler.seed_manager.has_active_request_seed.return_value = True
            elif invalid == 'shape':
                logits.shape = (16, 124160)
            rows = {'width': 3, 'width-48': 48, 'width-128': 128}.get(invalid, 16)
            if rows != 16:
                logits.shape = (1, 1, rows, 124160)
            with self.assertRaises(ValueError):
                sample_rows(sampler, logits, rows, operations)
            sampler.sample.assert_not_called()
            operations.pad.assert_not_called()

    def tiles(self, ids=None):
        """A 64-row block's two sampler tiles: the slices and the ids each sampler call returns."""
        sampler, operations = self.fixture()
        ids = ids or [SimpleNamespace(shape=(1, 1, 32, 1), name=name) for name in ('first', 'second')]
        sampler.sample = Mock(side_effect=[(tile, None) for tile in ids])
        slices = []

        def slice_rows(value, start, stop):
            slices.append(SimpleNamespace(shape=(1, 1, stop[2] - start[2], value.shape[3]), start=start, stop=stop))
            return slices[-1]

        operations.slice = Mock(side_effect=slice_rows)
        operations.concat = Mock(return_value='joined')
        return sampler, operations, ids, slices

    def test_sixty_four_rows_run_the_pinned_sampler_once_per_tile_and_join_the_ids(self):
        """M3: the sampler stays the plugin's 32-row tile; the 64-row block is two tiles."""
        for native in (False, True):
            sampler, operations, ids, slices = self.tiles()
            sampler.tt_sampling.vocab_size = sampler.tt_sampling.padded_vocab_size = 248320
            sampler._penalties_active = False
            logits = SimpleNamespace(shape=(1, 1, 64, 124160))
            self.assertEqual(sample_rows(sampler, logits, 64, operations, native_rows=native), 'joined')
            self.assertEqual([(piece.start, piece.stop) for piece in slices],
                             [((0, 0, 0, 0), (1, 1, 32, 124160)), ((0, 0, 32, 0), (1, 1, 64, 124160))])
            self.assertEqual([call.args[0] for call in sampler.sample.call_args_list], slices)
            self.assertTrue(all(call.kwargs == dict(enable_trace=False) for call in sampler.sample.call_args_list))
            operations.concat.assert_called_once_with(ids, dim=2)
            operations.pad.assert_not_called()
            self.assertEqual({id(call.args[0]) for call in operations.deallocate.call_args_list},
                             {id(value) for value in [*slices, *ids]}, 'the tiles and their ids are released once joined')
            self.assertEqual(operations.deallocate.call_count, 4)

    def test_a_sampler_that_reuses_its_output_across_tiles_is_refused(self):
        shared = SimpleNamespace(shape=(1, 1, 32, 1), name='shared')
        sampler, operations, ids, slices = self.tiles(ids=[shared, shared])
        with self.assertRaises(ValueError):
            sample_rows(sampler, SimpleNamespace(shape=(1, 1, 64, 124160)), 64, operations)
        operations.concat.assert_not_called()
        self.assertEqual({id(call.args[0]) for call in operations.deallocate.call_args_list},
                         {id(value) for value in [*slices, shared]})

    def test_an_id_layout_without_one_row_axis_is_refused(self):
        for shape in ((1, 32, 32), (16,), (1, 1, 32, 32)):
            ids = [SimpleNamespace(shape=shape, name=name) for name in ('first', 'second')]
            sampler, operations, ids, slices = self.tiles(ids=ids)
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                sample_rows(sampler, SimpleNamespace(shape=(1, 1, 64, 124160)), 64, operations)
            operations.concat.assert_not_called()
        first, second = SimpleNamespace(shape=(32,), name='first'), SimpleNamespace(shape=(1, 32), name='second')
        sampler, operations, ids, slices = self.tiles(ids=[first, second])
        with self.assertRaises(ValueError):
            sample_rows(sampler, SimpleNamespace(shape=(1, 1, 64, 124160)), 64, operations)
        operations.concat.assert_not_called()

    def test_a_failed_second_tile_releases_the_first_tile_and_its_ids(self):
        sampler, operations, ids, slices = self.tiles()
        sampler.sample = Mock(side_effect=[(ids[0], None), RuntimeError('device')])
        with self.assertRaises(RuntimeError):
            sample_rows(sampler, SimpleNamespace(shape=(1, 1, 64, 124160)), 64, operations)
        self.assertEqual({id(call.args[0]) for call in operations.deallocate.call_args_list},
                         {id(value) for value in [*slices, ids[0]]})
        operations.concat.assert_not_called()

    def test_a_tile_with_log_probabilities_is_refused(self):
        sampler, operations, ids, slices = self.tiles()
        sampler.sample = Mock(side_effect=[(ids[0], None), (ids[1], 'logprobs')])
        with self.assertRaises(ValueError):
            sample_rows(sampler, SimpleNamespace(shape=(1, 1, 64, 124160)), 64, operations)
        operations.concat.assert_not_called()

    def test_sampler_failure_releases_owned_padding(self):
        sampler, operations = self.fixture()
        sampler.sample.side_effect = RuntimeError('device')
        with self.assertRaises(RuntimeError):
            sample_rows(sampler, SimpleNamespace(shape=(1, 1, 2, 124160)), 2, operations)
        operations.deallocate.assert_called_once_with('padded')

    def test_padding_view_never_deallocates_original_storage(self):
        sampler, operations = self.fixture()
        operations.get_device_tensors = lambda value: [SimpleNamespace(buffer_address=lambda: 123)] * 2
        sample_rows(sampler, SimpleNamespace(shape=(1, 1, 2, 124160)), 2, operations)
        operations.deallocate.assert_not_called()

    def test_native_shape_need_not_support_slicing(self):
        class Shape:
            def __iter__(self):
                return iter((1, 1, 2, 124160))

            def __getitem__(self, index):
                raise TypeError('Native shape does not support slices')

        sampler, operations = self.fixture()
        self.assertEqual(sample_rows(sampler, SimpleNamespace(shape=Shape()), 2, operations), 'ids')

    def test_native_rows_preserve_input_width_without_padding(self):
        for rows in (1, 2, 4, 8, 16, 32):
            sampler, operations = self.fixture()
            sampler.tt_sampling.vocab_size = sampler.tt_sampling.padded_vocab_size = 248320
            sampler._penalties_active = False
            logits = SimpleNamespace(shape=(1, 1, rows, 124160))
            self.assertEqual(sample_rows(sampler, logits, rows, operations, native_rows=True), 'ids')
            sampler.sample.assert_called_once_with(logits, enable_trace=False)
            operations.pad.assert_not_called()
            operations.deallocate.assert_not_called()

    def test_native_rows_reject_unqualified_sampler_contracts(self):
        for invalid in ('selection', 'vocab', 'padding', 'penalties', 'logprobs', 'shard'):
            sampler, operations = self.fixture()
            sampler.tt_sampling.vocab_size = sampler.tt_sampling.padded_vocab_size = 248320
            sampler._penalties_active = sampler._log_probs_active = False
            logits = SimpleNamespace(shape=(1, 1, 1, 124160))
            if invalid == 'vocab':
                sampler.tt_sampling.vocab_size = 248319
            elif invalid == 'padding':
                sampler.tt_sampling.padded_vocab_size = 248352
            elif invalid == 'penalties':
                sampler._penalties_active = True
            elif invalid == 'logprobs':
                sampler._log_probs_active = True
            elif invalid == 'shard':
                logits.shape = (1, 1, 1, 248320)
            with self.assertRaises(ValueError):
                sample_rows(sampler, logits, 1, operations, native_rows=1 if invalid == 'selection' else True)
            sampler.sample.assert_not_called()
            operations.pad.assert_not_called()
