from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dspark_64k_shared_qk import shared_factory_scope, validate_shared


class SharedIntegrationTests(unittest.TestCase):
    def test_scope_order_and_restoration(self):
        order, records = [], []

        class Arm:
            operations = None

            @contextmanager
            def install(self):
                order.append('mlp_enter')
                yield
                order.append('mlp_exit')

        @contextmanager
        def shared(operations, admission):
            order.append('shared_enter')
            audit = dict(loads=[dict(rows=16, programs=3, retained_preparation_buffers=2)] * 48)
            yield audit
            order.append('shared_exit')
            audit.update(restored=True, released=True)

        module = SimpleNamespace(FusedT16Arm=Arm)
        with patch.dict(os.environ, QWEN_64K_SHARED_QK_AUDIT='1', QWEN_64K_MLP_AUDIT='1',
                QWEN_DSPARK_SFPU_REQUEST_SCREEN='1', QWEN_DSPARK_SFPU_TIMED='0'):
            with shared_factory_scope(module, shared, {}, records):
                with module.FusedT16Arm().install():
                    order.append('request')
        self.assertEqual(order, ['shared_enter', 'mlp_enter', 'request', 'mlp_exit', 'shared_exit'])
        self.assertIs(module.FusedT16Arm, Arm)
        self.assertEqual(len(records), 1)

    def test_missing_execution_rejected(self):
        with self.assertRaises(ValueError):
            validate_shared(dict(restored=True, released=True, loads=[]))
