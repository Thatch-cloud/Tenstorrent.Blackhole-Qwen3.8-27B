"""A step with no new request must reach the stock path, not the admission contract.

Run 35718626867 (v73) got all four users to a first token - the first run to do so - and
then died:

    ValueError: Fast serving requires one complete fresh prefill:
      prefill_slot=None new=[] cached=['cmpl-9629ed6b...'] spec={'cmpl-9629ed6b...': [...]}

prefill_slot None, no new requests, one cached request carrying speculative tokens: a
decode step, judged by a clause that exists to vet an ADMISSION. It reached that clause
because self.hook was None - _release_request had torn the hook down once every request
the lifecycle TRACKS had finished, while an untracked one was still decoding - so
neither of the two delegating branches above caught it.

These tests build a real FastServingLifecycle and call _execute. The module imports
without torch, unlike test_serving_lifecycle, so this runs on the CPU lane too.

The narrow fix is not the root cause. v73 adopted only two of four users onto the fast
path (see the propose/step record counts), and that is its own task; but refusing a step
that was never an admission is wrong independently of how many users are adopted.
"""

from pathlib import Path
import sys
import types
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).parent))

import serving_lifecycle
from serving_lifecycle import FastServingLifecycle


def scheduled(new=(), cached=(), spec=None, tokens=16, finished=()):
    return types.SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=list(cached)),
        scheduled_spec_decode_tokens=dict(spec or {}),
        total_num_scheduled_tokens=tokens,
        finished_req_ids=set(finished),
        has_structured_output_requests=False,
        scheduled_encoder_inputs={})


def lifecycle(delegated, hook=None, request_id=None, decoding_ids=()):
    built = FastServingLifecycle.__new__(FastServingLifecycle)
    built.failed = built.closed = built.prefill_pending = False
    built.request_id = request_id
    built.decoding_id = decoding_ids[-1] if decoding_ids else None
    built.decoding_ids = list(decoding_ids)
    built.capture = None
    built.hook = hook
    built.original_execute = lambda step: delegated.append(step) or 'stock'
    return built


class DelegateNonAdmissionStepsTests(unittest.TestCase):
    def test_the_v73_step_reaches_the_stock_path(self):
        """The exact shape that killed run 35718626867."""
        delegated = []
        built = lifecycle(delegated)
        step = scheduled(cached=['cmpl-9629ed6b'], spec={'cmpl-9629ed6b': [-1, -1]})
        self.assertEqual(built._execute(None, step), 'stock')
        self.assertEqual(delegated, [step])

    def test_it_raised_before_the_fix(self):
        """The negative control, expressed as the clause's own precondition: with a
        new request present the admission contract still applies and still refuses a
        step that also carries cached work."""
        built = lifecycle([])
        with self.assertRaisesRegex(ValueError, 'one complete fresh prefill'):
            built._execute(None, scheduled(new=[types.SimpleNamespace(req_id='b')],
                                           cached=['a']))

    def test_a_clean_admission_is_still_vetted_not_delegated(self):
        """The regression guard: exactly one new request and nothing else must still
        take the fast path's own admission route, not the stock one."""
        delegated = []
        built = lifecycle(delegated)
        # Reaching the admission body needs more of the object than this fixture
        # provides, so the assertion is that it did NOT delegate - the contract was
        # entered rather than bypassed.
        try:
            built._execute(None, scheduled(new=[types.SimpleNamespace(req_id='a')]))
        except Exception:
            pass
        self.assertEqual(delegated, [])

    def test_the_marker_names_the_request_and_fires_once_per_id(self):
        """A delegation that quietly swallows steps is how a fault becomes invisible."""
        logged = []
        loguru = types.ModuleType('loguru')
        loguru.logger = types.SimpleNamespace(info=lambda msg: logged.append(msg))
        built = lifecycle([])
        with unittest.mock.patch.dict(sys.modules, {'loguru': loguru}):
            for _ in range(3):
                built._execute(None, scheduled(cached=['cmpl-9629ed6b']))
            built._execute(None, scheduled(cached=['cmpl-other']))
        self.assertEqual(len(logged), 2, logged)
        self.assertIn('cmpl-9629ed6b', logged[0])
        self.assertIn('cmpl-other', logged[1])

    def test_a_zero_token_step_still_delegates(self):
        """The pre-existing branch above must keep working."""
        delegated = []
        built = lifecycle(delegated)
        step = scheduled(tokens=0)
        self.assertEqual(built._execute(None, step), 'stock')
        self.assertEqual(delegated, [step])

    def test_the_gate_is_published_by_the_property(self):
        """Guards the other half of the concurrency fix in the same module."""
        built = FastServingLifecycle.__new__(FastServingLifecycle)
        built.request_id = 'cmpl-held'
        self.assertEqual(serving_lifecycle.prefill_gate().held, 'cmpl-held')
        built.request_id = None
        self.assertIsNone(serving_lifecycle.prefill_gate().held)


if __name__ == '__main__':
    unittest.main()
