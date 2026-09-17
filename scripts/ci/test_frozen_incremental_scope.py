import unittest
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from frozen_incremental_scope import validate_records, runtime_scope


class IncrementalAdmissionTests(unittest.TestCase):
    def test_isolated_candidate_and_restoration(self):
        active = []
        observed = []
        route = Mock()
        variants = SimpleNamespace(validate_route=route)
        history = SimpleNamespace(StableHistoryKV=object())
        publication = SimpleNamespace()

        @contextmanager
        def incremental(history_class, publication_module, records):
            self.assertIs(history_class, history.StableHistoryKV)
            self.assertIs(publication_module, publication)
            active.append(True)
            try:
                yield
            finally:
                active.pop()
                records.append(dict(initial_position=32768, capacity=33024, failed=False,
                    restored=True, prepared=12, committed=11, discarded=1, max_touched_rows=64))

        def measure(**kwargs):
            observed.append((kwargs['gdn_shared_qk'], bool(active)))
            return {}

        full = SimpleNamespace(measure_dspark_request=measure)
        modules = dict(full_dspark_request=full, dspark_stable_history=history,
            dspark_publication_scope=publication, gdn_shared_qk_variants=variants)
        environment = dict(QWEN_FROZEN_COMBINED_RUNTIME='1', QWEN_CARDS_ALLOCATED='1',
            QWEN_DSPARK_REQUEST_CONTEXT='32768')
        options = dict(captured_publication=True, fused_t16_mlp=True,
            target_attention_t16=True, score_layout=True)
        with patch.dict(os.environ, environment, clear=True), patch.dict('sys.modules', modules), \
                patch('frozen_incremental_scope.qualify'), \
                patch('frozen_incremental_scope.incremental_history', incremental):
            with runtime_scope('.'):
                control = full.measure_dspark_request(**options)
                candidate = full.measure_dspark_request(**options, gdn_shared_qk=True)
                variants.validate_route(control, 'control')
                variants.validate_route(candidate, 'publication')
                with self.assertRaises(ValueError):
                    variants.validate_route(control, 'publication')
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(variants.validate_route, route)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with runtime_scope('.'):
                    raise RuntimeError('injected')
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(variants.validate_route, route)
        self.assertEqual(observed, [(True, False), (True, True)])
        self.assertFalse(active)
        self.assertEqual([call.args[1] for call in route.call_args_list], ['publication', 'publication'])

    def test_unallocated_hardware_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            with runtime_scope('.'):
                pass

    def test_complete_bounded_session(self):
        record = dict(initial_position=32768, capacity=33024, failed=False, restored=True,
            prepared=12, committed=11, discarded=1, max_touched_rows=64)
        validate_records([record], True)
        validate_records([], False)
        for field, value in (('failed', True), ('restored', False), ('initial_position', 65536),
                ('capacity', 66560), ('committed', 0), ('prepared', 11), ('max_touched_rows', 128)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records([dict(record, **{field: value})], True)
        with self.assertRaises(ValueError):
            validate_records([record], False)
        with self.assertRaises(ValueError):
            validate_records([], True)


if __name__ == '__main__':
    unittest.main()
