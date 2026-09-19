from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dflash_t32_preload import validate_fusion_routes


class FusionPreloadTests(unittest.TestCase):
    def test_current_width_routes_are_available(self):
        validate_fusion_routes(Path(__file__).parent)

    def test_legacy_missing_candidate_fails_before_model_loading(self):
        legacy = SimpleNamespace(__file__=str(Path(__file__).with_name('fused_t16_scope.py')),
            FusedT16Arm=type('LegacyArm', (), {}))
        with patch.dict('sys.modules', fused_t16_scope=legacy), self.assertRaisesRegex(ValueError, 'before loading'):
            validate_fusion_routes(Path(__file__).parent)

    def test_other_checkout_is_not_admitted(self):
        with self.assertRaises(ValueError):
            validate_fusion_routes(Path(__file__).parent / 'other')
