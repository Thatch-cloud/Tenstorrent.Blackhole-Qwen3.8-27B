from unittest.mock import patch
import unittest

import dflash_t16_native_scope as scope
from draft_attention_branch import prepare_attention_branch


class NativeT16ScopeTests(unittest.TestCase):
    def test_record_cannot_claim_exact_or_omit_source_admission(self):
        admission = dict(reports={str(context): digest for context, digest in scope.REPORTS.items()},
            sources=scope.hashes(scope.Path(scope.__file__).parent, scope.SOURCES),
            native_sources={**dict.fromkeys(scope.NATIVE_SOURCES, 'a' * 64), **scope.ORIGINAL},
            approximate_proposals=True, target_attention_changed=False)
        scope.validate_record(admission)
        for name, value in (('approximate_proposals', False), ('target_attention_changed', True),
                ('sources', {}), ('reports', {}), ('native_sources', {})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                scope.validate_record({**admission, name: value})

    def test_unadmitted_branch_fails_before_any_upload(self):
        with self.assertRaisesRegex(ValueError, 'admission'):
            prepare_attention_branch(None, None, {}, {}, None, native_head_layout=True,
                block_rows=16, native_proposal_attention=True)

    def test_scope_restores_after_failure_and_rejects_nesting(self):
        admission = dict(reports={}, sources={}, native_sources={})
        with patch.object(scope, 'admit', return_value=admission) as validate:
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                with scope.scoped_native_t16('evidence', 'sources', 'runtime'):
                    self.assertIs(scope.require_active(), admission)
                    with self.assertRaisesRegex(ValueError, 'Nested'):
                        with scope.scoped_native_t16('evidence', 'sources', 'runtime'):
                            self.fail('Nested scope entered')
                    raise RuntimeError('request failed')
            self.assertEqual(validate.call_count, 2)
        with self.assertRaisesRegex(ValueError, 'admission'):
            scope.require_active()

    def test_source_drift_fails_closed_and_clears_scope(self):
        with patch.object(scope, 'admit', side_effect=[{'sources': 'before'}, {'sources': 'after'}]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                with scope.scoped_native_t16('evidence', 'sources', 'runtime'):
                    pass
        with self.assertRaises(ValueError):
            scope.require_active()


if __name__ == '__main__':
    unittest.main()
