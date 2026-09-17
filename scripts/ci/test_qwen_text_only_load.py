import os
import unittest
from unittest.mock import patch

from qwen_text_only_load import text_only_load


class TextOnlyLoadTests(unittest.TestCase):
    def model_class(self):
        class Model:
            def __init__(self):
                self.vision_model = None
                self.text_weights = object()
            def init_vision_model(self):
                self.vision_model = object()
        return Model

    def test_explicit_scope_preserves_text_and_restores_vision_constructor(self):
        model_class, records = self.model_class(), []
        model = model_class()
        weights = model.text_weights
        original = model_class.init_vision_model
        with patch.dict(os.environ, QWEN_TEXT_ONLY_LOAD='1'):
            with text_only_load(model_class, records):
                model.init_vision_model()
                self.assertIsNone(model.vision_model)
                self.assertIs(model.text_weights, weights)
        self.assertIs(model_class.init_vision_model, original)
        self.assertTrue(records[0]['restored'])
        self.assertEqual(records[0]['skipped_vision_initializations'], 1)
        model.init_vision_model()
        self.assertIsNotNone(model.vision_model)

    def test_unselected_and_unused_scopes_fail(self):
        with patch.dict(os.environ, QWEN_TEXT_ONLY_LOAD='0'):
            with self.assertRaises(ValueError):
                with text_only_load(self.model_class(), []):
                    self.fail('Unselected scope entered')
        with patch.dict(os.environ, QWEN_TEXT_ONLY_LOAD='1'):
            with self.assertRaises(ValueError):
                with text_only_load(self.model_class(), []):
                    pass

    def test_explicit_vision_request_fails_and_binding_restores(self):
        model_class = self.model_class()
        original = model_class.init_vision_model
        with patch.dict(os.environ, QWEN_TEXT_ONLY_LOAD='1'):
            with self.assertRaises(ValueError):
                with text_only_load(model_class, []):
                    model_class().init_vision_model(reference_visual=object())
        self.assertIs(model_class.init_vision_model, original)
