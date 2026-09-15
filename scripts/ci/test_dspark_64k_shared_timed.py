from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dspark_64k_mlp_timed as mlp
from dspark_64k_shared_timed import factory_scope, identity_scope, SCREEN_RUN


class SharedTimingTests(unittest.TestCase):
    def test_composition_and_restoration(self):
        records, order = [], []

        class Arm:
            operations = None

            @contextmanager
            def install(self):
                order.append('mlp')
                yield
                order.append('mlp_restored')

        @contextmanager
        def shared(operations, admission):
            result = dict(loads=[dict(rows=16, programs=3, retained_preparation_buffers=2)] * 48)
            yield result
            result.update(restored=True, released=True)
            order.append('shared_released')

        module = SimpleNamespace(FusedT16Arm=Arm)
        original_run = mlp.SCREEN_RUN
        with identity_scope():
            self.assertEqual(mlp.SCREEN_RUN, SCREEN_RUN)
        self.assertEqual(mlp.SCREEN_RUN, original_run)
        with patch.object(mlp.splitk, 'require_timed'), patch.dict(os.environ,
                QWEN_64K_SHARED_QK_TIMED='1', QWEN_64K_MLP_TIMED='1', QWEN_64K_SHARED_QK_AUDIT='0'):
            with factory_scope(module, shared, {}, records):
                with module.FusedT16Arm().install():
                    order.append('request')
        self.assertIs(module.FusedT16Arm, Arm)
        self.assertEqual(order, ['mlp', 'request', 'mlp_restored', 'shared_released'])
        self.assertEqual(len(records), 1)
