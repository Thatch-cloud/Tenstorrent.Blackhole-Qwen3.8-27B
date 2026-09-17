import unittest

from gdn_multitoken import HASHES, HANDOFF_HASHES
from t32_commit_gate import GENERATED, validate


class T32CommitGateTests(unittest.TestCase):
    def setUp(self):
        self.sources = {'fixture': 'source-hash'}
        self.report = dict(backend='ttsim', rows=32, last_stage={'stage': 'complete'},
            model_adapter_checks=33, continuation_checks=33, precommit_unchanged_checks=33,
            stale_controls=1, native_hashes=HASHES, handoff_runtime_hashes=HANDOFF_HASHES,
            generated_hashes=GENERATED, commit_sources=self.sources, commit_sources_after=self.sources)
        for flag in ('passed', 'norm_gate', 'convolution', 'batched_convolution', 'dma_windows',
                'packed_checkpoints', 'continuation_enabled', 'compact_prologue', 'norm_batch_layer',
                'deferred_conv_publication', 'commit_only_gdn'):
            self.report[flag] = True

    def test_complete_component_is_not_full_request_admission(self):
        self.assertFalse(validate(self.report, self.sources)['full_request_qualified'])

    def test_partial_or_old_width_fails(self):
        for key, value in (('rows', 16), ('continuation_checks', 32),
                ('precommit_unchanged_checks', 32), ('model_adapter_checks', 32), ('stale_controls', 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate({**self.report, key: value}, self.sources)

    def test_changed_sources_fail(self):
        with self.assertRaises(ValueError):
            validate(self.report, {'fixture': 'changed'})


if __name__ == '__main__':
    unittest.main()
