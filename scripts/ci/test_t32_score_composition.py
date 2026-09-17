import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from t32_score_composition import DELTAS, RUNTIME, load, validate_dependencies, validate_score


class ScoreCompositionTests(unittest.TestCase):
    def test_report_digest_clean_exit_and_runtime_are_all_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
                sources={'fixture': 'a'}, sources_after={'fixture': 'a'})
            raw = json.dumps(report).encode()
            path = root / 'report.json'
            path.write_bytes(raw)
            path.with_suffix('.exit-status').write_text('0')
            (root / 'simulator-runtime.txt').write_text(RUNTIME)
            checksum = hashlib.sha256(raw).hexdigest()
            self.assertEqual(load(path, checksum), report)
            for target, bad, original in ((path, raw + b'\n', raw),
                    (path.with_suffix('.exit-status'), b'124', b'0'),
                    (root / 'simulator-runtime.txt', b'other', RUNTIME.encode())):
                target.write_bytes(bad)
                with self.assertRaises(ValueError):
                    load(path, checksum)
                target.write_bytes(original)

    def test_only_exact_reviewed_dependency_delta_allowed(self):
        before = {name: original for name, (original, unused) in DELTAS.items() if original is not None}
        after = {name: updated for name, (unused, updated) in DELTAS.items()}
        before['unchanged.py'] = after['unchanged.py'] = 'a' * 64
        validate_dependencies(before, after)
        for bad in (dict(after, **{'unchanged.py': 'b' * 64}),
                dict(after, **{'dspark_t32_prepared.py': 'b' * 64}),
                {name: value for name, value in after.items() if name != 'dspark_t32_score_layout.py'}):
            with self.assertRaises(ValueError):
                validate_dependencies(before, bad)

    def fixture(self):
        return dict(vocabulary=64, proposals=31, score_layout='fused', target_integrated=False,
            native_arithmetic_reference=True, native_sources={'native': 'a'}, native_sources_after={'native': 'a'},
            eager_checks=[dict(pattern=pattern, step=step, chip=chip, token_exact=True, full_vocabulary_exact=True)
                for pattern in range(3) for step in range(31) for chip in (0, 1)],
            replay_checks=[dict(repetition=repetition, step=step, chip=chip, token_and_scores_exact=True, bindings_stable=True)
                for repetition in range(4) for step in range(31) for chip in (0, 1)],
            input_checks=[dict(exact=True) for unused in range(28)],
            weight_checks=[dict(exact=True) for unused in range(8)],
            stale_controls=[dict(missing_update_detected=True) for unused in range(2)])

    def test_query_matrix_not_just_counts_and_pass_flag(self):
        report = self.fixture()
        validate_score(report)
        for field in ('eager_checks', 'replay_checks'):
            bad = copy.deepcopy(report)
            bad[field][-1] = bad[field][0]
            with self.assertRaises(ValueError):
                validate_score(bad)
        bad = copy.deepcopy(report)
        bad['weight_checks'][0]['exact'] = False
        with self.assertRaises(ValueError):
            validate_score(bad)


if __name__ == '__main__':
    unittest.main()
