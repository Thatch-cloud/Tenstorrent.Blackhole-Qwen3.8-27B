from types import SimpleNamespace
import unittest

from dspark_audit_observer import observe_audit


class AuditObserverTests(unittest.TestCase):
    def test_execution_and_restoration_including_failure(self):
        for fail in (False, True):
            calls, records = [], []
            module = SimpleNamespace(execute=lambda: calls.append('eager'))

            class Prepared:
                audit = True
                trace = 9
                device = SimpleNamespace(position=65536)
                operations = SimpleNamespace(execute_trace=lambda: calls.append('replay'))

                def update(self):
                    calls.append('update')

                def snapshot(self):
                    calls.append('snapshot')

                def read_tokens(self):
                    calls.append('tokens')
                    return (1, 2)

                def propose(self):
                    self.update()
                    module.execute()
                    self.snapshot()
                    self.operations.execute_trace()
                    if fail:
                        raise RuntimeError('original failure')
                    self.snapshot()
                    return self.read_tokens()

            class Publication:
                position = 65536

                def publish(self):
                    return 'published'

            module.PreparedDSparkProposal = Prepared
            original = Prepared.propose
            replay = Prepared.operations.execute_trace
            with observe_audit(module, Publication, records.append):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'original failure'):
                        Prepared().propose()
                else:
                    self.assertEqual(Prepared().propose(), (1, 2))
                self.assertEqual(Publication().publish(), 'published')
            self.assertIs(Prepared.propose, original)
            self.assertIs(Prepared.operations.execute_trace, replay)
            self.assertEqual(calls, ['update', 'eager', 'snapshot', 'replay'] +
                ([] if fail else ['snapshot', 'tokens']))
            totals = [entry for entry in records if entry['phase'] == 'proposal_total']
            self.assertEqual(totals[0]['status'], 'started')
            self.assertIsNone(totals[0]['elapsed_ms'])
            self.assertEqual(totals[-1]['status'], 'failed' if fail else 'completed')
            self.assertTrue(all(not entry['performance_qualified'] for entry in records))
