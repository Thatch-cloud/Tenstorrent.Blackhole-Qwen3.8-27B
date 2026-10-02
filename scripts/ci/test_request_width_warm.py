"""request_width_warm: the any-request engine's rows 1/2/4 eager warm-up, run before the packed blocks capture."""

from contextlib import contextmanager
from types import SimpleNamespace
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import force_argmax  # noqa: E402
import gdn_commit_dma  # noqa: E402
import gdn_multitoken_conv  # noqa: E402
import request_width_warm  # noqa: E402
import verifier_engine  # noqa: E402

LAYERS = 3


class Tensor:
    def __init__(self, label):
        self.label = label


class Operations:
    """A recording stand-in for ttnn: every tensor it hands out is tracked until it is deallocated."""

    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def __init__(self):
        self.events, self.live = [], []

    def new(self, label):
        tensor = Tensor(label)
        self.live.append(tensor)
        return tensor

    def from_torch(self, value, **options):
        self.events.append(('from_torch', tuple(value.shape)))
        return self.new('feature')

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim)

    def copy(self, source, destination):
        pass

    def deallocate(self, tensor):
        self.live.remove(tensor)
        self.events.append(('deallocate', tensor.label))

    def synchronize_device(self, mesh):
        self.events.append(('synchronize',))

    def begin_trace_capture(self, *args, **options):
        raise AssertionError('the warm-up captures no trace')


class Helper:
    def __init__(self, operations, events):
        self.operations, self.events = operations, events

    def allocate(self):
        return [self.operations.new('snapshot')]

    def save(self, snapshot):
        self.events.append(('save',))

    def restore(self, snapshot):
        self.events.append(('restore',))


class Fixture:
    def __init__(self, operations, rows, log):
        self.operations, self.rows, self.log = operations, rows, log
        self.retained = None
        if rows > 1:
            state = SimpleNamespace(entry=[], gdn=SimpleNamespace(rec_state=1, conv_states=[2]))
            self.retained = SimpleNamespace(records=[(state, {'states': 3, 'packed_conv_states': [4]}, [5])])

    def run(self, sharded_logits):
        self.log.append(('run', self.rows, sharded_logits))
        return self.operations.new('logits')

    def close(self):
        self.log.append(('fixture_close', self.rows))


class WarmTests(unittest.TestCase):
    def setUp(self):
        self.operations = Operations()
        self.events = self.operations.events
        self.helpers = [Helper(self.operations, self.events) for _ in range(LAYERS)]
        self.model = SimpleNamespace(mesh_device=SimpleNamespace(num_program_cache_entries=lambda: 700), layers=[0] * 62)
        self.sampler = object()
        self.batches, self.lines, self.captures = [], [], []

    def model_batch(self, model, tokens, start, pages, helpers, checkpoints, prefix, **options):
        self.batches.append((len(tokens), start, tuple(pages.shape), len(checkpoints), prefix, options))
        return Fixture(self.operations, len(tokens), self.events)

    def sample(self, sampler, logits, rows, operations, *, native_rows=False):
        self.events.append(('sample', rows, native_rows))
        return self.operations.new('ids')

    def prepare(self, mesh, layers, prefix):
        self.events.append(('prepare', prefix))
        return lambda prefix=prefix: self.events.append(('publish', prefix))

    @contextmanager
    def patched(self, sample=None):
        outer = self

        class Features:
            def __init__(self, model, taps, destinations, **options):
                outer.captures.append((tuple(taps), len(destinations)))

            @contextmanager
            def capture(self):
                outer.events.append(('tap_begin',))
                yield
                outer.events.append(('tap_end',))

            def close(self):
                outer.events.append(('tap_close',))

        def release(operations, tensors):
            for tensor in {id(value): value for value in tensors}.values():
                operations.deallocate(tensor)

        with patch.object(request_width_warm, 'ModelBatch', self.model_batch), \
                patch.object(force_argmax, 'sample_rows', sample or self.sample), \
                patch.object(gdn_commit_dma, 'prepare', self.prepare), \
                patch.object(gdn_multitoken_conv, 'addresses', lambda operations, tensor: (id(tensor),)), \
                patch.object(gdn_multitoken_conv, 'release_owned', release), \
                patch('prepared_target_features.PreparedTargetFeatures', Features):
            yield

    def warm(self, page_width=4096, sample=None, **options):
        with self.patched(sample):
            request_width_warm.warm_request_widths(self.operations, self.model, self.helpers, self.sampler, page_width,
                log=lambda template, *values: self.lines.append(template.format(*values)), **options)

    def test_every_width_is_sampled_native_and_fenced_in_order(self):
        self.warm()
        samples = [(index, event) for index, event in enumerate(self.events) if event[0] == 'sample']
        self.assertEqual([event for index, event in samples], [('sample', 1, True), ('sample', 2, True), ('sample', 4, True)])
        for index, event in samples:
            self.assertIn(('synchronize',), self.events[index + 1:index + 3], 'each sample is followed by a synchronize')
        self.assertEqual([batch[0] for batch in self.batches], [1, 2, 4])
        self.assertEqual([('run', rows, True) in self.events for rows in (1, 2, 4)], [True] * 3)

    def test_the_forward_runs_under_the_target_taps_capture_on_scratch_tensors(self):
        self.warm()
        from dflash_request_runtime import TARGET_TAPS
        self.assertEqual(self.captures, [(tuple(TARGET_TAPS), len(TARGET_TAPS))] * 3)
        begins = [i for i, event in enumerate(self.events) if event == ('tap_begin',)]
        self.assertEqual(len(begins), 3)
        for begin, rows in zip(begins, (1, 2, 4)):
            self.assertEqual(self.events[begin + 1], ('run', rows, True))
            self.assertEqual(self.events[begin + 2], ('tap_end',))
        self.assertIn(('from_torch', (1, 1, 4, 5120)), self.events)

    def test_every_publication_prefix_runs_once_for_rows_above_one_and_never_for_one_row(self):
        self.warm()
        self.assertEqual([e for e in self.events if e[0] == 'publish'],
                         [('publish', p) for p in range(3)] + [('publish', p) for p in range(5)])

    def test_every_scratch_allocation_is_released_and_the_state_is_restored(self):
        self.warm()
        self.assertEqual(self.operations.live, [], 'a scratch tensor survived the warm-up')
        self.assertEqual(self.events.count(('save',)), LAYERS)
        self.assertEqual(self.events.count(('restore',)), LAYERS * 4, 'before each width and once after the last')
        self.assertEqual(self.events.count(('tap_close',)), 3)
        self.assertEqual([e for e in self.events if e[0] == 'fixture_close'], [('fixture_close', r) for r in (1, 2, 4)])
        self.assertEqual([e for e in self.events if e[0] != 'deallocate'][-1], ('synchronize',), 'fenced before the last release')

    def test_a_failure_part_way_still_releases_everything(self):
        def failing(sampler, logits, rows, operations, *, native_rows=False):
            if rows == 2:
                raise RuntimeError('sampler failed')
            return self.operations.new('ids')
        with self.assertRaisesRegex(RuntimeError, 'sampler failed'):
            self.warm(sample=failing)
        self.assertEqual(self.operations.live, [])

    def test_no_trace_is_captured_and_the_marker_names_the_widths_and_programs(self):
        self.warm()
        self.assertNotIn('begin_trace_capture', [e[0] for e in self.events])
        self.assertEqual(len(self.lines), 1)
        self.assertTrue(self.lines[0].startswith('[PINDIAG] request widths warmed before the packed traces: '
                                                 'rows=(1, 2, 4) programs=700->700'), self.lines[0])

    def test_the_page_table_is_the_plugin_width_and_the_position_is_inside_it(self):
        self.warm()
        self.assertEqual({(batch[1], batch[2]) for batch in self.batches}, {(4096, (1, 4096))})
        with self.assertRaisesRegex(ValueError, 'inside the page table'):
            self.warm(page_width=64)
        with self.assertRaisesRegex(ValueError, 'Supported request widths'):
            self.warm(widths=(3,))


class ModelBatchDriftGuard(unittest.TestCase):
    """The warm-up's ModelBatch is exactly the one VerifierEngine.fixture builds for an any-request engine
    (serving_request_factory: norm_batch=True, attention_replay=not sequential=False, replay_group_rows=4, commit_only_gdn=True)."""

    @staticmethod
    def engine():
        engine = verifier_engine.VerifierEngine.__new__(verifier_engine.VerifierEngine)
        engine.model, engine.position, engine.pages, engine.helpers = 'model', 4096, 'pages', ['helper']
        engine.norm_batch, engine.attention_replay, engine.attention_mask_once = True, False, False
        engine.replay_group_rows, engine.short_context, engine.attention_audit = 4, False, False
        engine.commit_only_gdn = True
        return engine

    def test_the_keywords_equal_what_the_engine_fixture_passes(self):
        recorded = []
        with patch.object(verifier_engine, 'ModelBatch', lambda *args, **options: recorded.append((args, options))):
            for rows in (1, 2, 4):
                self.engine().fixture(rows, ['checkpoint'], retain=rows > 1, position=4096)
        self.assertEqual(len(recorded), 3)
        for rows, (args, options) in zip((1, 2, 4), recorded):
            with self.subTest(rows=rows):
                self.assertEqual(request_width_warm.request_fixture_options(rows, rows > 1), options)
                self.assertEqual(args, ('model', [1] * rows, 4096, 'pages', ['helper'], ['checkpoint'], 0 if rows == 1 else rows))

    def test_the_warm_up_passes_those_keywords_and_the_same_positional_shape(self):
        case = WarmTests('test_every_width_is_sampled_native_and_fenced_in_order')
        case.setUp()
        case.warm()
        self.assertEqual(len(case.batches), 3)
        for rows, position, shape, checkpoints, prefix, options in case.batches:
            with self.subTest(rows=rows):
                self.assertEqual(options, request_width_warm.request_fixture_options(rows, rows > 1))
                self.assertEqual((position, checkpoints, prefix), (4096, LAYERS, 0 if rows == 1 else rows))
                self.assertIs(options['retain_records'], rows > 1)
                self.assertEqual('commit_only_gdn' in options, rows > 1)


if __name__ == '__main__':
    unittest.main()
