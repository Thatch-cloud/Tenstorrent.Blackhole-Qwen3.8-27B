import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from frozen_draft_tail_stage import phase_adapter
from frozen_request_phases import IDENTITY, SCHEDULE, finish, prepare, qualify, validate_audits


class RequestPhaseTests(unittest.TestCase):
    def requests(self):
        block = dict(position=32768, rows=16, input_tokens=[1] * 16, accepted=1, committed=2)
        request = dict(length=32768, prompt_tokens=[1] * 32768, emitted=[2, 3, 4],
            committed_decode_tokens=2, exact=True, state_exact=True, inactive_exact=True,
            instrumented_timing=True, commit_only_gdn=True, blocks=[block],
            dspark=dict(native_attention=True, proposal_trace=True, proposal_checks=[
                dict(position=32768, exact=True, tensors=6), dict(position=32768, exact=True, tensors=6)]),
            gdn_verify_checks=[dict(position=32768, rows=16, unchanged=True)])
        return [dict(copy.deepcopy(request), arm=arm) for arm in ('control', 'publication')]

    def fixture(self):
        report = {name: {'fixture': name} for name in IDENTITY}
        report.update(passed=True, closed_cleanly=True, checkpoint_closed=True, stage='complete',
            request_phase='audit', correctness_only=True, pp=None, committed_tg=None,
            request_checks=self.requests(), workflow_run='qualification-run')
        report.update(sources_after=report['sources'], native_sources_after=report['native_sources'])
        return report

    def test_all_audit_invariants_and_routes(self):
        calls = []
        validate_audits(self.requests(), lambda value, arm: calls.append(arm))
        self.assertEqual(calls, ['control', 'publication'])
        for field, value in (('exact', False), ('instrumented_timing', False), ('gdn_verify_checks', []),
                ('committed_decode_tokens', 1), ('emitted', [99]), ('blocks', [])):
            requests = self.requests()
            requests[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_audits(requests, lambda *args: None)

    def test_partial_timeout_and_identity_changes_rejected(self):
        current = self.fixture()
        self.assertEqual(len(qualify(current, current, lambda *args: None)), 2)
        for field, value in (('closed_cleanly', False), ('passed', False), ('checkpoint_closed', False),
                ('stage', 'full_request_2_control_timed'), ('sources_after', {}), ('committed_tg', 100)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualify(dict(current, **{field: value}), current, lambda *args: None)
        for name in IDENTITY:
            with self.subTest(identity=name), self.assertRaises(ValueError):
                qualify(current, dict(current, **{name: 'changed'}), lambda *args: None)

    def test_absent_configuration_preserves_combined_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            phase = prepare(directory, {}, SCHEDULE)
            self.assertEqual(phase['schedule'], SCHEDULE)
            self.assertFalse(finish(phase, {}))

    def test_audit_phase_never_creates_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'frozen-request-phase.json').write_text('{"mode":"audit"}')
            report = dict(request_checks=self.requests(), stages=[dict(elapsed_seconds=70)])
            phase = prepare(directory, report, SCHEDULE)
            self.assertEqual(phase['schedule'], SCHEDULE[:2])
            module = SimpleNamespace(validate_route=lambda *args: None)
            with patch.dict(sys.modules, gdn_shared_qk_variants=module):
                self.assertTrue(finish(phase, report))
            self.assertTrue(report['correctness_only'])
            self.assertIsNone(report['committed_tg'])
            report['stages'] = [dict(elapsed_seconds=101)]
            with self.assertRaisesRegex(ValueError, '100 seconds'):
                prepare(directory, report, SCHEDULE)

    def test_timed_phase_requires_pinned_clean_audits_and_four_fresh_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            retained = self.fixture()
            payload = json.dumps(retained).encode()
            Path(directory, 'frozen-request-qualification.json').write_bytes(payload)
            configuration = dict(mode='timed', qualification_sha256=hashlib.sha256(payload).hexdigest())
            Path(directory, 'frozen-request-phase.json').write_text(json.dumps(configuration))
            current = {key: retained[key] for key in IDENTITY}
            module = SimpleNamespace(validate_route=lambda *args: None)
            with patch.dict(sys.modules, gdn_shared_qk_variants=module):
                phase = prepare(directory, current, SCHEDULE)
                self.assertEqual(phase['schedule'], SCHEDULE[2:])
                current['request_checks'] = [dict(arm=arm, instrumented_timing=audit)
                    for arm, audit in SCHEDULE[2:]]
                self.assertFalse(finish(phase, current))
                self.assertEqual(len(current['request_checks']), 6)
                self.assertEqual(current['retained_audit']['fresh_timed_requests'], 4)
                with self.assertRaises(ValueError):
                    finish(phase, current)
                Path(directory, 'frozen-request-qualification.json').write_bytes(payload + b' ')
                with self.assertRaisesRegex(ValueError, 'digest'):
                    prepare(directory, current, SCHEDULE)

    def test_unknown_schedule_or_phase_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'frozen-request-phase.json')
            path.write_text('{"mode":"audit"}')
            with self.assertRaises(ValueError):
                prepare(directory, {}, SCHEDULE[:2])
            path.write_text('{"mode":"skip"}')
            with self.assertRaises(ValueError):
                prepare(directory, {}, SCHEDULE)

    def test_real_orchestrator_adapter_is_anchored(self):
        source = Path(__file__).with_name('dspark_request_experiment.py').read_text()
        changed = phase_adapter(source)
        self.assertIn('request_phase = prepare(', changed)
        self.assertIn('if finish(request_phase, report):', changed)
        with self.assertRaises(ValueError):
            phase_adapter(changed)


if __name__ == '__main__':
    unittest.main()
