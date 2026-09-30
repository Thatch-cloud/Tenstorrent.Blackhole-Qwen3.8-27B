import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from contextlib import ExitStack, contextmanager

from serving_startup import model_walk, recipe_paths, start, stop, weight_streams
from test_serving_fast_policy import FastPolicyTests


class StartupTests(unittest.TestCase):
    def test_explicit_paths_match_loaded_runtime(self):
        config = FastPolicyTests().fixture()
        with TemporaryDirectory() as directory:
            config.additional_config['qwen_fast_runtime'] = {
                name: directory for name in ('directory', 'runtime_root', 'fixtures', 'target_snapshot')}
            with patch.dict(os.environ, {'TT_METAL_HOME': directory}):
                paths = recipe_paths(config)
                self.assertEqual(paths['runtime_root'], Path(directory).resolve())
            with patch.dict(os.environ, {'TT_METAL_HOME': str(Path(directory) / 'wrong')}):
                with self.assertRaises(ValueError):
                    recipe_paths(config)

    def test_no_implicit_runtime_or_fixture_defaults(self):
        config = FastPolicyTests().fixture()
        with self.assertRaises(ValueError):
            recipe_paths(config)
        config.additional_config['qwen_fast_runtime'] = {'directory': '.'}
        with self.assertRaises(ValueError):
            recipe_paths(config)

    def test_start_cannot_replace_an_existing_owner(self):
        worker = SimpleNamespace(_qwen_fast_resources=object())
        with self.assertRaises(ValueError):
            start(worker)

    def test_stop_releases_once_and_retains_owner_on_failure(self):
        resources = Mock()
        worker = SimpleNamespace(_qwen_fast_resources=resources, _qwen_fast_attachment=object())
        resources.close.side_effect = RuntimeError('cleanup failed')
        with self.assertRaises(RuntimeError):
            stop(worker)
        self.assertIs(worker._qwen_fast_resources, resources)
        resources.close.side_effect = None
        stop(worker)
        stop(worker)
        self.assertEqual(resources.close.call_count, 2)
        self.assertIsNone(worker._qwen_fast_resources)


class WeightStreamTests(unittest.TestCase):
    """serving_startup.weight_streams: the serial MLP block stream by default, skipped only
    where serving_runtime.register_reader_reason admits the register-epilogue reader."""

    M3 = {'QWEN_FAST_SKIP_BLOCK_STREAM': '1', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}

    def run_streams(self, environ, requests=4, built=3):
        # `built` of the three layers still hold a packed w_gate_up (the C1 model graft
        # leaves it None).
        layers = [SimpleNamespace(feed_forward=SimpleNamespace(weights=SimpleNamespace(
            w_gate_up='wgu%d' % index if index < built else None))) for index in range(3)]
        model = SimpleNamespace(mesh_device='mesh', layers=layers)
        events = []

        @contextmanager
        def owned_streams(operations, mesh, weights):
            events.append(('enter', operations, mesh, list(weights)))
            yield 'streams', 'weight-pool'
            events.append('exit')

        policy = Mock(return_value=dict(scheduler_requests=requests))
        resources = ExitStack()
        flags = ('QWEN_FAST_SKIP_BLOCK_STREAM', 'QWEN_FAST_PACKED_STEP', 'QWEN_FAST_FOUR_AS_TWO',
                 'QWEN_FAST_SINGLE_GATEUP')
        clean = {name: value for name, value in os.environ.items() if name not in flags}
        with patch.dict(os.environ, dict(clean, **environ), clear=True), \
                patch('dflash_device.pindiag') as marker:
            result = weight_streams(resources, 'ttnn', model, owned_streams, policy, Path('/recipe'))
        resources.close()
        return result, events, marker

    def test_every_flag_unset_enters_the_stream_exactly_as_before(self):
        for environ in ({}, {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}):
            with self.subTest(environ=environ):
                (block_stream, pool), events, marker = self.run_streams(environ)
                self.assertEqual(block_stream, dict(evidence=Path('/recipe') / 'block-stream-evidence', streams='streams'))
                self.assertEqual(pool, 'weight-pool')
                self.assertEqual(events, [('enter', 'ttnn', 'mesh', ['wgu0', 'wgu1', 'wgu2']), 'exit'])
                marker.assert_not_called()

    def test_the_skip_flag_at_the_sixty_four_row_block_builds_no_stream_and_says_so(self):
        (block_stream, pool), events, marker = self.run_streams(self.M3)
        self.assertEqual((block_stream, pool, events), (None, None, []))
        marker.assert_called_once_with('[PINDIAG] block stream skipped for {}: {}', 'the 64-row block',
                                       'register-epilogue reader on native w_gate_up')
        self.assertEqual(marker.call_args.args[0].format(*marker.call_args.args[1:]),
                         '[PINDIAG] block stream skipped for the 64-row block: register-epilogue reader on native w_gate_up')

    def test_the_skip_flag_keeps_the_stream_at_every_other_shape_and_says_so(self):
        # A run with the flag set that builds the stream anyway must not pass for an A1
        # measurement: the NOT-skipped marker names the shape that was configured.
        cases = ((self.M3, 2, 'users=2 FOUR_AS_TWO=0 PACKED_STEP=1'),
                 (self.M3, 1, 'users=1 FOUR_AS_TWO=0 PACKED_STEP=1'),
                 (dict(self.M3, QWEN_FAST_FOUR_AS_TWO='1'), 4, 'users=4 FOUR_AS_TWO=1 PACKED_STEP=1'),
                 ({'QWEN_FAST_SKIP_BLOCK_STREAM': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}, 4,
                  'users=4 FOUR_AS_TWO=0 PACKED_STEP=unset'))
        for environ, requests, shape in cases:
            with self.subTest(environ=environ, requests=requests):
                (block_stream, pool), events, marker = self.run_streams(environ, requests)
                self.assertEqual(pool, 'weight-pool')
                self.assertEqual(len(events), 2)
                marker.assert_called_once()
                self.assertEqual(marker.call_args.args[0].format(*marker.call_args.args[1:]),
                                 '[PINDIAG] block stream NOT skipped: QWEN_FAST_SKIP_BLOCK_STREAM=1 but ' + shape)

    C1 = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}

    def test_the_single_gate_up_flag_at_the_sixty_four_row_block_counts_what_the_model_holds(self):
        (block_stream, pool), events, marker = self.run_streams(self.C1, built=0)
        self.assertEqual((block_stream, pool, events), (None, None, []))
        self.assertEqual(marker.call_args.args[0].format(*marker.call_args.args[1:]),
                         '[PINDIAG] block stream skipped for the single gate/up copy: w_gate_up present on 0 of 3 layers')

    def test_the_single_gate_up_flag_refuses_a_model_that_still_holds_w_gate_up(self):
        # The model graft not applied: the stream would be skipped and the marker would
        # claim C1 ran while every layer kept its packed copy.
        for built in (1, 3):
            with self.subTest(built=built), \
                    self.assertRaisesRegex(ValueError, 'w_gate_up is present on %d of 3 layers' % built):
                self.run_streams(self.C1, built=built)

    def test_the_single_gate_up_flag_is_refused_at_every_other_shape(self):
        for environ, requests in ((self.C1, 1), (self.C1, 2), (dict(self.C1, QWEN_FAST_FOUR_AS_TWO='1'), 4),
                                  ({'QWEN_FAST_SINGLE_GATEUP': '1'}, 4)):
            with self.subTest(environ=environ, requests=requests), \
                    self.assertRaisesRegex(ValueError, 'admitted only at the 64-row M3 block'):
                self.run_streams(environ, requests, built=0)


class ModelWalkTests(unittest.TestCase):
    def test_p0_names_the_weights_the_model_state_and_the_kv_caches(self):
        weights = [SimpleNamespace(w1='w1-%d' % index, w2='w2-%d' % index, w3='w3-%d' % index, w_gate_up=None)
                   for index in range(2)]
        layers = [SimpleNamespace(feed_forward=SimpleNamespace(weights=weight)) for weight in weights]
        model = SimpleNamespace(layers=layers, tt_ccl='ccl', sampling='sampler', embd='embedding',
                                lm_head_weight='head', _deltanet_external_states='states', _gdn_prefill_scratch='scratch',
                                _paged_kv_caches='paged', rope='rope')
        walk = model_walk(SimpleNamespace(kv_caches='kv'), model)
        self.assertEqual(walk['target.mlp.w1_w3'], ['w1-0', 'w1-1', 'w3-0', 'w3-1'])
        self.assertEqual(walk['target.mlp.w2'], ['w2-0', 'w2-1'])
        self.assertEqual(walk['target.mlp.w_gate_up'], [None, None])
        self.assertIs(walk['target.layers.other'][0], layers[0])
        self.assertEqual(walk['model.tt_ccl_and_sampler'], ['ccl', 'sampler'])
        self.assertEqual((walk['model.embedding'], walk['model.lm_head']), ('embedding', 'head'))
        self.assertEqual(walk['model.gdn_states_and_scratch'], ['states', 'scratch', None, None])
        self.assertEqual(walk['kv_caches'], ['kv', 'paged'])
        self.assertEqual(walk['model.other'], {'rope': 'rope'})
        # The layers reference the model's shared TT_CCL and the KV caches; the specific
        # categories must claim those buffers before the generic per-layer walk does.
        order = list(walk)
        for specific in ('target.mlp.w_gate_up', 'model.tt_ccl_and_sampler', 'kv_caches', 'model.gdn_states_and_scratch'):
            self.assertLess(order.index(specific), order.index('target.layers.other'))
        self.assertEqual(order[-1], 'model.other')


class StartLedgerTests(unittest.TestCase):
    """start() end to end on fakes: P0 and P1 are recorded only with
    QWEN_FAST_MEMORY_LEDGER=1, and the skip branch's recipe (None) reaches the attach."""

    def run_start(self, environ):
        import sys
        import memory_ledger

        weights = SimpleNamespace(w1='w1', w2='w2', w3='w3', w_gate_up='wgu')
        model = SimpleNamespace(mesh_device='mesh', args=SimpleNamespace(vocab_size=1000),
                                layers=[SimpleNamespace(feed_forward=SimpleNamespace(weights=weights))])
        runner = SimpleNamespace(model=SimpleNamespace(model=[model]), kv_caches=[])
        worker = SimpleNamespace(vllm_config='config', model_runner=runner)
        seen = {}

        @contextmanager
        def owned_streams(operations, mesh, tensors):
            seen['streams'] = list(tensors)
            yield 'streams', 'weight-pool'

        @contextmanager
        def attach(worker, operations, **options):
            seen['attach'] = options
            yield dict(lifecycle='lifecycle')

        recorded, begun = [], []

        def begin(operations, probe, **options):
            begun.append((operations, probe))
            if not memory_ledger.enabled():
                return None
            memory_ledger._active = SimpleNamespace()
            return memory_ledger._active

        def record(phase, point=None, **walked):
            if memory_ledger.active() is not None:
                recorded.append((phase, sorted(walked)))

        flags = ('QWEN_FAST_MEMORY_LEDGER', 'QWEN_FAST_SKIP_BLOCK_STREAM', 'QWEN_FAST_PACKED_STEP',
                 'QWEN_FAST_FOUR_AS_TWO', 'QWEN_FAST_SINGLE_GATEUP')
        clean = {name: value for name, value in os.environ.items() if name not in flags}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'generation_config.json').write_text('{"eos_token_id": 7}')
            paths = dict(directory=root, runtime_root=root, fixtures=root, target_snapshot=root)
            modules = {'ttnn': SimpleNamespace(name='ttnn'),
                       'sampling_link_policy': SimpleNamespace(DESCRIPTOR='descriptor', SOURCES={}),
                       'full_dflash_request': SimpleNamespace(load_dflash_fixtures=lambda path: 'fixtures'),
                       'mlp_block_stream_pool': SimpleNamespace(owned_streams=owned_streams)}
            with patch.dict(os.environ, dict(clean, TT_MESH_GRAPH_DESC_PATH=str(root / 'descriptor'), **environ),
                            clear=True), \
                    patch.dict(sys.modules, modules), \
                    patch('serving_startup.recipe_paths', return_value=paths), \
                    patch('serving_startup.validate_fast_config', return_value=dict(scheduler_requests=4)), \
                    patch('serving_runtime.attach_combined_runtime', side_effect=attach), \
                    patch.object(memory_ledger, 'begin', side_effect=begin), \
                    patch.object(memory_ledger, 'record', side_effect=record), \
                    patch('dflash_device.pindiag'):
                try:
                    start(worker)
                    self.assertEqual(worker._qwen_fast_attachment['lifecycle'], 'lifecycle')
                    stop(worker)
                finally:
                    memory_ledger._active = None
        return seen, recorded, begun, worker

    def test_the_ledger_is_silent_and_the_stream_unchanged_by_default(self):
        seen, recorded, begun, worker = self.run_start({})
        self.assertEqual((recorded, begun), ([], []))
        self.assertEqual(seen['streams'], ['wgu'])
        self.assertEqual(seen['attach']['block_stream']['streams'], 'streams')
        self.assertIsNone(worker._qwen_fast_attachment)

    def test_the_ledger_records_p0_and_p1_around_the_stream_when_asked(self):
        seen, recorded, begun, worker = self.run_start({'QWEN_FAST_MEMORY_LEDGER': '1'})
        self.assertEqual(begun, [(begun[0][0], 'w2')])
        self.assertEqual([phase for phase, _ in recorded], ['P0', 'P1'])
        self.assertIn('target.mlp.w_gate_up', recorded[0][1])
        self.assertEqual(recorded[1][1], ['block_stream', 'weight_pool'])

    def test_the_skip_branch_hands_the_attach_no_stream(self):
        seen, recorded, begun, worker = self.run_start({'QWEN_FAST_SKIP_BLOCK_STREAM': '1', 'QWEN_FAST_PACKED_STEP': '1',
                                                        'QWEN_FAST_FOUR_AS_TWO': '0'})
        self.assertNotIn('streams', seen)
        self.assertIsNone(seen['attach']['block_stream'])
