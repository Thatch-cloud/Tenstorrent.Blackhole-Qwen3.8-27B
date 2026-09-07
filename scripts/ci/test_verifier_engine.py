from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from greedy_session import GreedySession
from verifier_engine import VerifierEngine, capture_widths


class EngineLifecycleTests(unittest.TestCase):
    def test_future_family_is_forwarded_to_the_model_fixture(self):
        engine = VerifierEngine.__new__(VerifierEngine)
        engine.model, engine.pages, engine.helpers = object(), object(), []
        engine.position, engine.norm_batch, engine.attention_replay = 4078, True, True
        with patch('verifier_engine.ModelBatch') as factory:
            engine.fixture(32, [], retain=True, position=4096)
        self.assertEqual(factory.call_args.args[2], 4096)
        self.assertIs(factory.call_args.kwargs['attention_replay'], True)

    def test_invalid_replay_selection_fails_before_request_access(self):
        with patch.dict(sys.modules, ttnn=SimpleNamespace()):
            for enabled, norm in ((True, False), ('true', True), (1, True)):
                with self.assertRaisesRegex(ValueError, 'replay attention'):
                    VerifierEngine(None, None, None, None, norm_batch=norm, attention_replay=enabled)

    def test_norm_batch_is_forwarded_to_warm_and_retained_fixtures(self):
        engine = VerifierEngine.__new__(VerifierEngine)
        engine.model, engine.pages, engine.helpers = object(), object(), []
        engine.position = 4095
        for enabled in (False, True):
            engine.norm_batch = enabled
            for rows in (1, 2, 4, 8, 16, 32):
                for retain in (False, True):
                    with patch('verifier_engine.ModelBatch') as factory:
                        engine.fixture(rows, [], retain=retain)
                    self.assertIs(factory.call_args.kwargs['norm_batch'], enabled)
                    self.assertIs(factory.call_args.kwargs['retain_records'], retain)

    def test_invalid_norm_batch_fails_before_request_or_device_access(self):
        with patch.dict(sys.modules, ttnn=SimpleNamespace()):
            for enabled in (None, 0, 1, 'true'):
                with self.assertRaisesRegex(ValueError, 'boolean'):
                    VerifierEngine(None, None, None, None, norm_batch=enabled)

    def test_capture_geometry_respects_budget_and_capacity_before_device_work(self):
        self.assertEqual(capture_widths(65535, 65536, 32, 1), (1,))
        self.assertEqual(capture_widths(65531, 65536, 32, 5), (1, 2, 4))
        self.assertEqual(capture_widths(4095, 65536, 32, 128), (1, 2, 4, 8, 16, 32))
        for geometry in ((65535, 65536, 32, 2), (-1, 65536, 32, 1),
                         (0, 65536, 32, 0), (0, 65536, 64, 1), (True, 65536, 32, 1)):
            with self.assertRaises(ValueError):
                capture_widths(*geometry)

    def test_constructor_failure_poisoning_prevents_device_state_retry(self):
        for fail in ('allocate', 'fixture'):
            session = GreedySession('request', [0, 1, 2], 0, vocab_size=100, max_new_tokens=33)
            helpers = [SimpleNamespace(live=[object()] * 5, allocate=Mock(return_value=[object()] * 5),
                                       save=Mock()) for index in range(48)]
            if fail == 'allocate':
                helpers[0].allocate.side_effect = RuntimeError('allocation failure')
            operations = SimpleNamespace(synchronize_device=Mock())
            with patch.dict(sys.modules, ttnn=operations), patch('verifier_engine.addresses', return_value=(1, 2)), \
                 patch('verifier_engine.release_owned'), patch.object(VerifierEngine, 'fixture', side_effect=RuntimeError('fixture failure')):
                with self.assertRaises(RuntimeError):
                    VerifierEngine(SimpleNamespace(mesh_device=object()), session, SimpleNamespace(shape=(1, 1024)), helpers)
            self.assertEqual(session.phase, 'failed')
            self.assertEqual(session.emitted, [0])
            with self.assertRaises(ValueError):
                session.propose('request')
            session.close('request')

    def fixture(self, rows=4):
        session = GreedySession('request', [0, 1, 2] * 12, 0, vocab_size=100, max_new_tokens=64)
        engine = VerifierEngine.__new__(VerifierEngine)
        engine.session, engine.position = session, session.position
        engine.phase, engine.pending = 'idle', None
        engine.mesh = object()
        engine.model = SimpleNamespace(args=SimpleNamespace(vocab_size=100))
        engine.operations = SimpleNamespace(execute_trace=Mock(return_value=None), synchronize_device=Mock(),
            get_device_tensors=lambda tensor: [tensor, tensor], to_torch=lambda tensor: tensor)
        engine.validate_bindings = Mock()
        retained = Mock() if rows > 1 else None
        if retained is not None:
            retained.replay.side_effect = lambda operation: operation()
            retained.commit.side_effect = lambda prefix, **kwargs: kwargs['publication'](prefix)
        engine.helpers = [SimpleNamespace(restore=Mock())]
        bucket = dict(first=True, fixture=SimpleNamespace(retained=retained), trace=7,
            output=(None, torch.tensor([1, 2, 0, 1])), commits={prefix: 20 + prefix for prefix in range(rows + 1)},
            checkpoints=[[object()]])
        engine.buckets = {rows: bucket}
        return engine, session, bucket

    def test_verified_output_is_not_committed_until_publication(self):
        engine, session, bucket = self.fixture()
        with patch('verifier_engine.stage_inputs'):
            ticket = session.propose('request', max_rows=4)
            predictions, timing = engine.verify(ticket)
            self.assertEqual(session.emitted, [0])
            self.assertEqual(engine.phase, 'verified')
            self.assertGreaterEqual(timing['input_ms'], 0)
            decision = session.commit('request', ticket, predictions, engine.publish)
            self.assertEqual(decision.emitted, (1, 2, 0, 1))
            self.assertEqual(session.committed_decode_tokens, 4)
            self.assertEqual(engine.position, session.position)
            self.assertEqual(engine.phase, 'idle')
            engine.operations.execute_trace.assert_called_with(engine.mesh, 24, cq_id=0, blocking=True)
            ticket = session.propose('request', max_rows=4)
            engine.verify(ticket)
            bucket['fixture'].retained.replay.assert_called_once()
            session.abort('request', ticket, engine.publish)
            self.assertEqual(engine.position, session.position)
            self.assertEqual(session.committed_decode_tokens, 4)

    def test_family_bucket_publication_uses_the_verified_bucket_key(self):
        from attention_request_plan import capture_plan
        engine, unused_session, bucket = self.fixture(rows=16)
        session = GreedySession('request', [index % 3 for index in range(4096)], 1, vocab_size=100,
            max_new_tokens=64, prefer_full_suffix=True)
        engine.session, engine.position = session, session.position
        engine.replay_plan = capture_plan(session.position, 65536, 32, 63)
        engine.buckets = {(16, 4352): bucket}
        bucket['output'] = (None, torch.tensor([(index + 2) % 3 for index in range(16)]))
        with patch('verifier_engine.stage_inputs'):
            ticket = session.propose('request', max_rows=16)
            self.assertEqual(len(ticket.tokens), 16)
            predictions, _ = engine.verify(ticket)
            self.assertEqual(engine.pending_key, (16, 4352))
            engine.replay_plan = SimpleNamespace(select=Mock(side_effect=AssertionError('must not reroute publication')))
            session.commit('request', ticket, predictions, engine.publish)
        self.assertIsNone(engine.pending_key)
        self.assertEqual(engine.position, session.position)
        bucket['fixture'].retained.commit.assert_called_once()

    def test_single_token_abort_restores_entry_without_advancing(self):
        engine, session, bucket = self.fixture(rows=1)
        with patch('verifier_engine.stage_inputs'):
            ticket = session.propose('request', max_rows=1)
            engine.verify(ticket)
            session.abort('request', ticket, engine.publish)
        engine.helpers[0].restore.assert_called_once_with(bucket['checkpoints'][0])
        self.assertEqual(session.committed_decode_tokens, 0)
        self.assertEqual(engine.position, 36)

    def test_failed_staging_or_execution_poison_request_without_accounting(self):
        for failure in ('stage', 'execute', 'binding'):
            engine, session, bucket = self.fixture()
            ticket = session.propose('request', max_rows=4)
            if failure == 'execute':
                engine.operations.execute_trace.side_effect = RuntimeError('device failure')
            if failure == 'binding':
                engine.validate_bindings.side_effect = RuntimeError('binding failure')
            with patch('verifier_engine.stage_inputs', side_effect=RuntimeError('stage failure') if failure == 'stage' else None):
                with self.assertRaises(RuntimeError):
                    engine.verify(ticket)
            self.assertEqual((engine.phase, session.phase), ('failed', 'failed'))
            self.assertEqual(session.emitted, [0])
            self.assertEqual(session.committed_decode_tokens, 0)
            with self.assertRaises(ValueError):
                session.commit('request', ticket, [1] * 4, engine.publish)

    def test_failed_publication_does_not_advance_host_history(self):
        engine, session, bucket = self.fixture()
        with patch('verifier_engine.stage_inputs'):
            ticket = session.propose('request', max_rows=4)
            predictions, _ = engine.verify(ticket)
        bucket['fixture'].retained.commit.side_effect = RuntimeError('publication failure')
        with self.assertRaises(RuntimeError):
            session.commit('request', ticket, predictions, engine.publish)
        self.assertEqual((engine.phase, session.phase), ('failed', 'failed'))
        self.assertEqual(session.emitted, [0])
        self.assertEqual(engine.position, 36)

    def test_cannot_publish_before_verification_or_close_pending_block(self):
        engine, session, bucket = self.fixture()
        with self.assertRaises(ValueError):
            engine.publish(1)
        with patch('verifier_engine.stage_inputs'):
            ticket = session.propose('request', max_rows=4)
            engine.verify(ticket)
            with self.assertRaises(ValueError):
                engine.close()
            with self.assertRaises(ValueError):
                engine.verify(ticket)
            with self.assertRaises(ValueError):
                engine.publish(1)
        session.abort('request', ticket, engine.publish)
